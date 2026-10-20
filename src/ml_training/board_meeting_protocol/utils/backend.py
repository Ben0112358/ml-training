class TextModel:
    """One prompt in, one text reply out. Model id may be ignored."""

    def complete(self, prompt: str, model: str | None, on_delta=None) -> str:
        raise NotImplementedError
