import pathlib


class CompressedDataset:
    def get_size(self):
        raise NotImplementedError("get_size() must be implemented in subclasses")

    def save(self, dir: pathlib.Path):
        raise NotImplementedError("save() must be implemented in subclasses")
