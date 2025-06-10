from dataset import Dataset

from .report import Report


class Evaluation:
    def evaluate(self, original: Dataset, restored: Dataset, labels) -> Report:
        t1 = self.execute(original, labels)
        t2 = self.execute(restored, labels)
        return self.merge(t1, t2)

    def execute(self, dataset: Dataset):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def merge(self, t1, t2) -> Report:
        raise NotImplementedError("This method should be implemented by subclasses.")
