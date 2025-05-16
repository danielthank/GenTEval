from report import Report

from dataset import Dataset


class Evaluation:
    def evaluate(self, original: Dataset, restored: Dataset) -> Report:
        t1 = self.execute(original)
        t2 = self.execute(restored)
        return self.merge(t1, t2)

    def execute(self, dataset: Dataset):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def merge(self, t1, t2) -> Report:
        raise NotImplementedError("This method should be implemented by subclasses.")
