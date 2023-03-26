import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


class DataAugmenter:
    def __init__(self) -> None:
        self.random_cutout_size = 0

        self.rotation_p = 0
        self.rotation_max_deg = 15

        self.jiggle_p = 0
        self.jiggle_max = 0.03

        self.mirror_p = 0.5

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.random_cutout_size:
            # It could be also done before first interpolation, but there would be no option
            # of caching the interpolation and preprocessing and augmentation would be mixed.
            frame_len = x.shape[0]
            size = random.randint(0, int(self.max_batch_dropout * frame_len))
            start_offset = random.randint(0, x.shape[1] - size)
            x = torch.cat([x[:start_offset, :, :], x[start_offset + size :, :, :]], dim=0)
            x = x.permute(1, 2, 0)
            x = F.interpolate(x, [frame_len], mode="linear")
            x = x.permute(2, 0, 1)

        if np.random.rand() < self.rotation_p:
            rot = torch.Tensor(
                R.from_euler(
                    "xyz",
                    np.random.uniform(-self.rotation_max_deg, self.rotation_max_deg, 3),
                    degrees=True,
                ).as_matrix()
            )
            x = x @ rot
        if np.random.rand() < self.jiggle_p:
            x += torch.tensor(
                np.random.uniform(-self.jiggle_max, self.jiggle_max, x.shape), dtype=x.dtype
            )
        if np.random.rand() < self.mirror_p:
            x[:, :, 0] = -x[:, :, 0]
        return x
