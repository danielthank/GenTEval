class Dataset:
    def __init__(self):
        self.traces = None

    def extend(self, other) -> "Dataset":
        if not isinstance(other, Dataset):
            raise TypeError("Can only extend with another Dataset instance")
        if self.traces is None:
            self.traces = other.traces
        else:
            self.traces.update(other.traces)
