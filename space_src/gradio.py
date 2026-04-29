class _Theme:
    def __call__(self, *args, **kwargs):
        return self


class _Themes:
    Soft = _Theme
    Citrus = _Theme


themes = _Themes()


class _Component:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def click(self, *args, **kwargs):
        return self

    def change(self, *args, **kwargs):
        return self


class _Context(_Component):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def queue(self, *args, **kwargs):
        return self

    def launch(self, *args, **kwargs):
        return self


Blocks = _Context
Row = _Context
Column = _Context
Accordion = _Context
Markdown = _Component
Image = _Component
Textbox = _Component
Slider = _Component
Dropdown = _Component
Checkbox = _Component
Button = _Component
Video = _Component
Number = _Component
File = _Component
Audio = _Component
Examples = _Component
