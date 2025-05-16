import pathlib


class Dataset:
    def __init__(self):
        self.spans = None
        self.labels = None
        self.traces = None

    def load_spans(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def load_labels(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def spans_to_traces(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def save(self, dir: pathlib.Path):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def load(self, dir: pathlib.Path):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def traces_to_spans(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_size(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
