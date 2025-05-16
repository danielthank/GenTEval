from evaluation import Evaluation
from evaluation import Dataset

class TraceRCAReport:
    pass

class TraceRCAEvaluation(Evaluation):
    def execute(self, dataset: Dataset):
        pass

    def merge(self, t1, t2) -> TraceRCAReport:
        pass
