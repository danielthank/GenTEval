import numpy as np
import torch
from ctgan import CTGAN
from ctgan.synthesizers.base import random_state


class GenTCTGAN(CTGAN):
    @random_state
    def sample(self, n, conditions=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if conditions is not None:
            global_condition_vec = np.zeros(
                (self._batch_size, self._data_sampler.dim_cond_vec()), dtype="float32"
            )
            for column, value in conditions.items():
                # TODO: how to check if the column is discrete?
                condition_info = self._transformer.convert_column_name_value_to_id(
                    column, value
                )
                id_ = self._data_sampler._discrete_column_cond_st[
                    condition_info["discrete_column_id"]
                ]
                id_ += condition_info["value_id"]
                global_condition_vec[:, id_] = 1
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)
