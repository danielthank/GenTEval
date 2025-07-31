from genteval.dataset import Dataset


class Evaluator:
    def evaluate(self, dataset: Dataset, labels):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def execute(self, dataset: Dataset):
        raise NotImplementedError("This method should be implemented by subclasses.")
