import pathlib
import pickle

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

from compressors.trace import Trace
from dataset import Dataset

from .config import GenTConfig


class StartTimeSynthesizer:
    def __init__(self, config: GenTConfig):
        self.config = config
        self.synthesizer = None
        self.graph_count = None

    def _get_start_time_dataset(self, dataset: Dataset):
        column_names = ["graph", "startTime"]
        start_time_dataset = []
        for trace in dataset.traces.values():
            trace = Trace(trace)
            graph = str(trace.edges)
            if graph == "[]":  # TODO: support graph with no edges
                continue
            start_time = trace.start_time
            start_time_dataset.append([graph, start_time])
        return pd.DataFrame(start_time_dataset, columns=column_names)

    def distill(self, dataset):
        start_time_dataset = self._get_start_time_dataset(dataset)

        self.graph_count = start_time_dataset["graph"].value_counts().to_dict()
        metadata = Metadata()
        metadata.add_table("transactions")
        metadata.add_column("graph", sdtype="categorical")
        metadata.add_column("startTime", sdtype="numerical")
        self.synthesizer = CTGANSynthesizer(
            metadata=metadata,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            generator_dim=self.config.generator_dim,
            discriminator_dim=self.config.discriminator_dim,
            verbose=True,
        )
        self.synthesizer.fit(start_time_dataset)

    def save(self, dir: pathlib.Path):
        self.synthesizer.save(dir / "start_time_synthesizer")
        pickle.dump(
            self.graph_count,
            open(dir / "graph_count.pkl", "wb"),
        )

    def load(self, dir: pathlib.Path):
        self.synthesizer = CTGANSynthesizer.load(dir / "start_time_synthesizer")
        self.graph_count = pickle.load(open(dir / "graph_count.pkl", "rb"))

    def synthesize(self):
        pass
