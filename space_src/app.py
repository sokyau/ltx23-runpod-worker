import os
import subprocess
import sys

# Disable torch.compile / dynamo before any torch import
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

try:
    import xformers  # noqa: F401
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "xformers==0.0.32.post2", "--no-build-isolation"], check=False)

# Clone LTX-2 repo at a pinned compatible commit and install packages
LTX_REPO_URL = "https://github.com/Lightricks/LTX-2.git"
LTX_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LTX-2")
LTX_COMMIT = "ae855f8538843825f9015a419cf4ba5edaf5eec2"

if not os.path.exists(LTX_REPO_DIR):
    print(f"Cloning {LTX_REPO_URL}...")
    subprocess.run(["git", "clone", LTX_REPO_URL, LTX_REPO_DIR], check=True)
else:
    print(f"Using existing repo at {LTX_REPO_DIR}")

print(f"Checking out commit {LTX_COMMIT}...")
subprocess.run(["git", "-C", LTX_REPO_DIR, "checkout", LTX_COMMIT], check=True)

print("Installing ltx-core and ltx-pipelines from pinned repo commit...")
subprocess.run(
    [
        sys.executable, "-m", "pip", "install",
        "--force-reinstall", "--no-deps",
        "-e", os.path.join(LTX_REPO_DIR, "packages", "ltx-core"),
        "-e", os.path.join(LTX_REPO_DIR, "packages", "ltx-pipelines"),
    ],
    check=True,
)

sys.path.insert(0, os.path.join(LTX_REPO_DIR, "packages", "ltx-pipelines", "src"))
sys.path.insert(0, os.path.join(LTX_REPO_DIR, "packages", "ltx-core", "src"))

import logging
import random
import tempfile
from pathlib import Path

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import spaces
import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number, decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils import euler_denoising_loop
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    combined_image_conditionings,
    denoise_video_only,
    encode_prompts,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video

# Force-patch xformers attention into the LTX attention module.
from ltx_core.model.transformer import attention as _attn_mod
print(f"[ATTN] Before patch: memory_efficient_attention={_attn_mod.memory_efficient_attention}")
try:
    from xformers.ops import memory_efficient_attention as _mea
    _attn_mod.memory_efficient_attention = _mea
    print(f"[ATTN] After patch: memory_efficient_attention={_attn_mod.memory_efficient_attention}")
except Exception as e:
    print(f"[ATTN] xformers patch FAILED: {type(e).__name__}: {e}")

logging.getLogger().setLevel(logging.INFO)

MAX_SEED = np.iinfo(np.int32).max
DEFAULT_PROMPT = (
    "An astronaut hatches from a fragile egg on the surface of the Moon, "
    "the shell cracking and peeling apart in gentle low-gravity motion. "
    "Fine lunar dust lifts and drifts outward with each movement, floating "
    "in slow arcs before settling back onto the ground."
)
DEFAULT_FRAME_RATE = 24.0

# Resolution presets: (width, height)
RESOLUTIONS = {
    "high": {"16:9": (1536, 1024), "9:16": (1024, 1536), "1:1": (1024, 1024)},
    "low": {"16:9": (768, 512), "9:16": (512, 768), "1:1": (768, 768)},
}


class LTX23DistilledA2VPipeline(DistilledPipeline):
    """DistilledPipeline with optional audio conditioning."""

    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        audio_path: str | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
    ):
        # Standard path when no audio input is provided.
        if audio_path is None:
            return super().__call__(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=enhance_prompt,
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        (ctx_p,) = encode_prompts(
            [prompt],
            self.model_ledger,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0].path if len(images) > 0 else None,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        video_duration = num_frames / frame_rate
        decoded_audio = decode_audio_from_file(audio_path, self.device, 0.0, video_duration)
        if decoded_audio is None:
            raise ValueError(f"Could not extract audio stream from {audio_path}")

        encoded_audio_latent = vae_encode_audio(decoded_audio, self.model_ledger.audio_encoder())
        audio_shape = AudioLatentShape.from_duration(batch=1, duration=video_duration, channels=8, mel_bins=16)
        expected_frames = audio_shape.frames
        actual_frames = encoded_audio_latent.shape[2]

        if actual_frames > expected_frames:
            encoded_audio_latent = encoded_audio_latent[:, :, :expected_frames, :]
        elif actual_frames < expected_frames:
            pad = torch.zeros(
                encoded_audio_latent.shape[0],
                encoded_audio_latent.shape[1],
                expected_frames - actual_frames,
                encoded_audio_latent.shape[3],
                device=encoded_audio_latent.device,
                dtype=encoded_audio_latent.dtype,
            )
            encoded_audio_latent = torch.cat([encoded_audio_latent, pad], dim=2)

        video_encoder = self.model_ledger.video_encoder()
        transformer = self.model_ledger.transformer()
        stage_1_sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=self.device)

        def denoising_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = combined_image_conditionings(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        video_state = denoise_video_only(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            initial_audio_latent=encoded_audio_latent,
        )

        torch.cuda.synchronize()
        cleanup_memory()

        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self.model_ledger.spatial_upsampler(),
        )
        stage_2_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, device=self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = combined_image_conditionings(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        video_state = denoise_video_only(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=stage_2_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=encoded_audio_latent,
        )

        torch.cuda.synchronize()
        del transformer
        del video_encoder
        cleanup_memory()

        decoded_video = vae_decode_video(
            video_state.latent,
            self.model_ledger.video_decoder(),
            tiling_config,
            generator,
        )
        original_audio = Audio(
            waveform=decoded_audio.waveform.squeeze(0),
            sampling_rate=decoded_audio.sampling_rate,
        )
        return decoded_video, original_audio


# Model repos
LTX_MODEL_REPO = "Lightricks/LTX-2.3"
GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"

# Download model checkpoints
print("=" * 80)
print("Downloading LTX-2.3 distilled model + Gemma...")
print("=" * 80)

checkpoint_path = hf_hub_download(repo_id=LTX_MODEL_REPO, filename="ltx-2.3-22b-distilled-1.1.safetensors")
spatial_upsampler_path = hf_hub_download(repo_id=LTX_MODEL_REPO, filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
gemma_root = snapshot_download(repo_id=GEMMA_REPO)

print(f"Checkpoint: {checkpoint_path}")
print(f"Spatial upsampler: {spatial_upsampler_path}")
print(f"Gemma root: {gemma_root}")

# Initialize pipeline WITH text encoder and optional audio support
pipeline = LTX23DistilledA2VPipeline(
    distilled_checkpoint_path=checkpoint_path,
    spatial_upsampler_path=spatial_upsampler_path,
    gemma_root=gemma_root,
    loras=[],
    quantization=QuantizationPolicy.fp8_cast(),
)

# Preload all models for ZeroGPU tensor packing.
print("Preloading all models (including Gemma and audio components)...")
ledger = pipeline.model_ledger
_transformer = ledger.transformer()
_video_encoder = ledger.video_encoder()
_video_decoder = ledger.video_decoder()
_audio_encoder = ledger.audio_encoder()
_audio_decoder = ledger.audio_decoder()
_vocoder = ledger.vocoder()
_spatial_upsampler = ledger.spatial_upsampler()
_text_encoder = ledger.text_encoder()
_embeddings_processor = ledger.gemma_embeddings_processor()

ledger.transformer = lambda: _transformer
ledger.video_encoder = lambda: _video_encoder
ledger.video_decoder = lambda: _video_decoder
ledger.audio_encoder = lambda: _audio_encoder
ledger.audio_decoder = lambda: _audio_decoder
ledger.vocoder = lambda: _vocoder
ledger.spatial_upsampler = lambda: _spatial_upsampler
ledger.text_encoder = lambda: _text_encoder
ledger.gemma_embeddings_processor = lambda: _embeddings_processor
print("All models preloaded (including Gemma text encoder and audio encoder)!")

print("=" * 80)
print("Pipeline ready!")
print("=" * 80)


def log_memory(tag: str):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        free, total = torch.cuda.mem_get_info()
        print(f"[VRAM {tag}] allocated={allocated:.2f}GB peak={peak:.2f}GB free={free / 1024**3:.2f}GB total={total / 1024**3:.2f}GB")


def detect_aspect_ratio(image) -> str:
    if image is None:
        return "16:9"
    if hasattr(image, "size"):
        w, h = image.size
    elif hasattr(image, "shape"):
        h, w = image.shape[:2]
    else:
        return "16:9"
    ratio = w / h
    candidates = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}
    return min(candidates, key=lambda k: abs(ratio - candidates[k]))


def on_image_upload(first_image, last_image, high_res):
    ref_image = first_image if first_image is not None else last_image
    aspect = detect_aspect_ratio(ref_image)
    tier = "high" if high_res else "low"
    w, h = RESOLUTIONS[tier][aspect]
    return gr.update(value=w), gr.update(value=h)


def on_highres_toggle(first_image, last_image, high_res):
    ref_image = first_image if first_image is not None else last_image
    aspect = detect_aspect_ratio(ref_image)
    tier = "high" if high_res else "low"
    w, h = RESOLUTIONS[tier][aspect]
    return gr.update(value=w), gr.update(value=h)


@spaces.GPU(duration=75)
@torch.inference_mode()
def generate_video(
    first_image,
    last_image,
    input_audio,
    prompt: str,
    duration: float,
    enhance_prompt: bool = True,
    seed: int = 42,
    randomize_seed: bool = True,
    height: int = 1024,
    width: int = 1536,
    progress=gr.Progress(track_tqdm=True),
):
    try:
        torch.cuda.reset_peak_memory_stats()
        log_memory("start")

        current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

        frame_rate = DEFAULT_FRAME_RATE
        num_frames = int(duration * frame_rate) + 1
        num_frames = ((num_frames - 1 + 7) // 8) * 8 + 1

        print(f"Generating: {height}x{width}, {num_frames} frames ({duration}s), seed={current_seed}")

        images = []
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        if first_image is not None:
            temp_first_path = output_dir / f"temp_first_{current_seed}.jpg"
            if hasattr(first_image, "save"):
                first_image.save(temp_first_path)
            else:
                temp_first_path = Path(first_image)
            images.append(ImageConditioningInput(path=str(temp_first_path), frame_idx=0, strength=1.0))

        if last_image is not None:
            temp_last_path = output_dir / f"temp_last_{current_seed}.jpg"
            if hasattr(last_image, "save"):
                last_image.save(temp_last_path)
            else:
                temp_last_path = Path(last_image)
            images.append(ImageConditioningInput(path=str(temp_last_path), frame_idx=num_frames - 1, strength=1.0))

        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        log_memory("before pipeline call")

        video, audio = pipeline(
            prompt=prompt,
            seed=current_seed,
            height=int(height),
            width=int(width),
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            audio_path=input_audio,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
        )

        log_memory("after pipeline call")

        output_path = tempfile.mktemp(suffix=".mp4")
        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )

        log_memory("after encode_video")
        return str(output_path), current_seed

    except Exception as e:
        import traceback
        log_memory("on error")
        print(f"Error: {str(e)}\n{traceback.format_exc()}")
        return None, current_seed


with gr.Blocks(title="LTX-2.3 Distilled") as demo:
    gr.Markdown("# LTX-2.3 F2LF: Fast Audio-Video Generation with Frame Conditioning")
    gr.Markdown(
        "Fast and high quality video + audio generation with first and last frame conditioning and optional audio input "
        "[[model]](https://huggingface.co/Lightricks/LTX-2.3) "
        "[[code]](https://github.com/Lightricks/LTX-2)"
    )

    with gr.Row():
        with gr.Column():
            with gr.Row():
                first_image = gr.Image(label="First Frame (Optional)", type="pil")
                last_image = gr.Image(label="Last Frame (Optional)", type="pil")
            input_audio = gr.Audio(label="Audio Input (Optional)", type="filepath")
            prompt = gr.Textbox(
                label="Prompt",
                info="for best results - make it as elaborate as possible",
                value="Make this image come alive with cinematic motion, smooth animation",
                lines=3,
                placeholder="Describe the motion and animation you want...",
            )
            duration = gr.Slider(label="Duration (seconds)", minimum=1.0, maximum=10.0, value=3.0, step=0.1)
                

            generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

            with gr.Accordion("Advanced Settings", open=False):
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, value=10, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                with gr.Row():
                    width = gr.Number(label="Width", value=1536, precision=0)
                    height = gr.Number(label="Height", value=1024, precision=0)
                with gr.Row():
                    enhance_prompt = gr.Checkbox(label="Enhance Prompt", value=False)
                    high_res = gr.Checkbox(label="High Resolution", value=True)

        with gr.Column():
            output_video = gr.Video(label="Generated Video", autoplay=True)

    gr.Examples(
        examples=[
            [
                None,
                "pinkknit.jpg",
                None,
                "The camera falls downward through darkness as if dropped into a tunnel. "
                "As it slows, five friends wearing pink knitted hats and sunglasses lean "
                "over and look down toward the camera with curious expressions. The lens "
                "has a strong fisheye effect, creating a circular frame around them. They "
                "crowd together closely, forming a symmetrical cluster while staring "
                "directly into the lens.",
                3.0,
                False,
                42,
                True,
                1024,
                1024,
            ],
        ],
        inputs=[
            first_image, last_image, input_audio, prompt, duration,
            enhance_prompt, seed, randomize_seed, height, width,
        ],
    )

    first_image.change(
        fn=on_image_upload,
        inputs=[first_image, last_image, high_res],
        outputs=[width, height],
    )

    last_image.change(
        fn=on_image_upload,
        inputs=[first_image, last_image, high_res],
        outputs=[width, height],
    )

    high_res.change(
        fn=on_highres_toggle,
        inputs=[first_image, last_image, high_res],
        outputs=[width, height],
    )

    generate_btn.click(
        fn=generate_video,
        inputs=[
            first_image, last_image, input_audio, prompt, duration, enhance_prompt,
            seed, randomize_seed, height, width,
        ],
        outputs=[output_video, seed],
    )


css = """
.fillable{max-width: 1200px !important}
"""

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Citrus(), css=css)
