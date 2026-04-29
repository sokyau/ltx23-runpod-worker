def GPU(fn=None, **_kwargs):
    """Compatibility shim for Hugging Face ZeroGPU decorators outside Spaces."""
    if callable(fn):
        return fn

    def decorator(inner):
        return inner

    return decorator
