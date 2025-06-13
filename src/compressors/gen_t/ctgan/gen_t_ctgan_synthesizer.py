import warnings

import pandas as pd
from sdv._utils import _groupby_list
from sdv.errors import ConstraintsNotMetError
from sdv.single_table import CTGANSynthesizer
from sdv.single_table.base import COND_IDX
from sdv.single_table.ctgan import _validate_no_category_dtype
from sdv.single_table.utils import detect_discrete_columns, handle_sampling_error

from .gen_t_ctgan import GenTCTGAN


class GenTCTGANSynthesizer(CTGANSynthesizer):
    # Override _fit to plug in GenTCTGAN
    def _fit(self, processed_data):
        """Fit the model to the table.

        Args:
            processed_data (pandas.DataFrame):
                Data to be learned.
        """
        _validate_no_category_dtype(processed_data)

        transformers = self._data_processor._hyper_transformer.field_transformers
        discrete_columns = detect_discrete_columns(
            self.metadata, processed_data, transformers
        )
        self._model = GenTCTGAN(**self._model_kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")
            self._model.fit(processed_data, discrete_columns=discrete_columns)

    # Override _sample to support conditional sampling
    def _sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        """
        if conditions is None:
            print(num_rows)
        else:
            print(num_rows, conditions.keys())
        """
        if conditions is None:
            return self._model.sample(num_rows)
        else:
            sampled = self._model.sample(num_rows, conditions=conditions)
            for column, value in conditions.items():
                if column in sampled.columns:
                    sampled[column] = value
            return sampled

    def _sample_with_conditions(
        self,
        conditions,
        max_tries_per_batch,
        batch_size,
        progress_bar=None,
        output_file_path=None,
        condition_columns=None,
    ):
        if condition_columns is None:
            return super()._sample_with_conditions(
                conditions,
                max_tries_per_batch,
                batch_size,
                progress_bar,
                output_file_path,
            )

        conditions.index.name = COND_IDX
        conditions = conditions.reset_index()
        grouped_conditions = conditions.groupby(_groupby_list(condition_columns))

        # sample
        all_sampled_rows = []

        for group, dataframe in grouped_conditions:
            if not isinstance(group, tuple):
                group = [group]

            condition = dict(zip(condition_columns, group))
            condition_df = dataframe.iloc[0].to_frame().T
            try:
                transformed_condition = self._data_processor.transform(
                    condition_df, is_condition=True
                )
            except ConstraintsNotMetError as error:
                raise ConstraintsNotMetError(
                    "Provided conditions are not valid for the given constraints."
                ) from error

            # Only choose the column that is in condition_columns
            transformed_condition = transformed_condition[
                transformed_condition.columns.intersection(condition_columns)
            ]
            transformed_conditions = pd.concat(
                [transformed_condition] * len(dataframe), ignore_index=True
            )
            transformed_columns = list(transformed_conditions.columns)
            if not transformed_conditions.empty:
                transformed_conditions.index = dataframe.index
                transformed_conditions[COND_IDX] = dataframe[COND_IDX]

            if len(transformed_columns) == 0:
                sampled_rows = self._conditionally_sample_rows(
                    dataframe=dataframe,
                    condition=condition,
                    transformed_condition=None,
                    max_tries_per_batch=max_tries_per_batch,
                    batch_size=batch_size,
                    progress_bar=progress_bar,
                    output_file_path=output_file_path,
                )
                all_sampled_rows.append(sampled_rows)
            else:
                transformed_groups = transformed_conditions.groupby(
                    _groupby_list(transformed_columns)
                )
                for transformed_group, transformed_dataframe in transformed_groups:
                    if not isinstance(transformed_group, tuple):
                        transformed_group = [transformed_group]

                    transformed_condition = dict(
                        zip(transformed_columns, transformed_group)
                    )
                    sampled_rows = self._conditionally_sample_rows(
                        dataframe=transformed_dataframe,
                        condition=condition,
                        transformed_condition=transformed_condition,
                        max_tries_per_batch=max_tries_per_batch,
                        batch_size=batch_size,
                        progress_bar=progress_bar,
                        output_file_path=output_file_path,
                    )

                    merged_rows = pd.merge(
                        dataframe,
                        sampled_rows,
                        on=transformed_columns + [COND_IDX],
                        suffixes=(None, "_remove"),
                        how="left",
                    )

                    # Remove the columns that has "_remove" suffix
                    merged_rows.drop(
                        merged_rows.filter(like="_remove").columns,
                        axis=1,
                        inplace=True,
                    )

                    all_sampled_rows.append(merged_rows)

        all_sampled_rows = pd.concat(all_sampled_rows)
        if len(all_sampled_rows) == 0:
            return all_sampled_rows

        all_sampled_rows = all_sampled_rows.reset_index().drop(columns=["index"])
        all_sampled_rows = all_sampled_rows.set_index(COND_IDX)
        all_sampled_rows.index.name = conditions.index.name
        all_sampled_rows = all_sampled_rows.sort_index()

        return all_sampled_rows

    def sample_remaining_columns(
        self,
        known_columns,
        max_tries_per_batch=100,
        batch_size=None,
        output_file_path=None,
        condition_columns=None,
        progress_bar=None,
    ):
        if condition_columns is None:
            return super().sample_remaining_columns(
                known_columns, max_tries_per_batch, batch_size, output_file_path
            )

        known_columns = known_columns.copy()
        sampled = pd.DataFrame()
        try:
            sampled = self._sample_with_conditions(
                known_columns,
                max_tries_per_batch,
                batch_size,
                progress_bar,
                output_file_path,
                condition_columns,
            )

        except (Exception, KeyboardInterrupt) as error:
            handle_sampling_error(output_file_path, error)

        return sampled

    def trim(self):
        self._data_processor._transformers_by_sdtype = None
        self._model._data_sampler._rid_by_cat_cols = None
