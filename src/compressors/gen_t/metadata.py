import pathlib

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

from compressors.trace import Trace
from dataset import Dataset

from .config import GenTConfig


class MetadataSynthesizer:
    def __init__(self, config: GenTConfig):
        self.config = config
        self.root_synthesizer = None
        self.chain_synthesizer = None

    def _get_sdv_metadata(self):
        metadata = Metadata()
        metadata.add_table("metadata")
        metadata.add_column("graph", sdtype="categorical")
        metadata.add_column("chain", sdtype="categorical")
        for i in range(self.config.chain_length):
            metadata.add_column(f"gapFromParent_{i}", sdtype="numerical")
            metadata.add_column(f"duration_{i}", sdtype="numerical")
        return metadata

    def _get_metadata_dataset(self, dataset: Dataset):
        root_dataset = []
        chain_dataset = []
        columns = ["graph", "chain"]
        for i in range(self.config.chain_length):
            columns.append(f"gapFromParent_{i}")
            columns.append(f"duration_{i}")

        for trace in dataset.traces.values():
            trace = Trace(trace)
            # TODO: support graph with no edges
            if len(trace) <= 1:
                continue
            for chain in trace.chains(self.config.chain_length):
                row = [
                    str(trace.edges),
                    "#".join(
                        [trace.unique_name(span_id) for span_id in chain["chain"]]
                    ),
                ]
                for span_id in chain["chain"]:
                    row.extend(
                        [trace.gap_from_parent(span_id), trace.duration(span_id)]
                    )
                if chain["is_root"]:
                    root_dataset.append(row)
                else:
                    chain_dataset.append(row)
        return {
            "root": pd.DataFrame(root_dataset, columns=columns),
            "chain": pd.DataFrame(chain_dataset, columns=columns),
        }

    def distill(self, dataset):
        metadata_dataset = self._get_metadata_dataset(dataset)
        sdv_metadata = self._get_sdv_metadata()
        self.root_synthesizer = CTGANSynthesizer(
            metadata=sdv_metadata,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            generator_dim=self.config.generator_dim,
            discriminator_dim=self.config.discriminator_dim,
            enforce_rounding=False,
            verbose=True,
        )
        self.root_synthesizer.fit(metadata_dataset["root"])
        self.chain_synthesizer = CTGANSynthesizer(
            metadata=sdv_metadata,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            generator_dim=self.config.generator_dim,
            discriminator_dim=self.config.discriminator_dim,
            enforce_rounding=False,
            verbose=True,
        )
        self.chain_synthesizer.fit(metadata_dataset["chain"])

    def save(self, dir: pathlib.Path):
        self.root_synthesizer.save(dir / "root_synthesizer")
        self.chain_synthesizer.save(dir / "chain_synthesizer")

    def load(self, dir: pathlib.Path):
        self.root_synthesizer = CTGANSynthesizer.load(dir / "root_synthesizer")
        self.chain_synthesizer = CTGANSynthesizer.load(dir / "chain_synthesizer")

    def synthesize(self):
        pass
