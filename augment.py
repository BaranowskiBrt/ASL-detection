import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class DataAugmenter:
    def __init__(self) -> None:
        self.rotation_p = 0
        self.rotation_max_deg = 15

        self.jiggle_p = 0
        self.jiggle_max = 0.01

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rot = torch.Tensor(
            R.from_euler(
                "zyx",
                np.random.uniform(-self.rotation_max_deg, self.rotation_max_deg, 3),
                degrees=True,
            ).as_matrix()
        )
        if np.random.rand() < self.rotation_p:
            x = x @ rot
        if np.random.rand() < self.jiggle_p:
            x += torch.tensor(
                np.random.uniform(-self.jiggle_max, self.jiggle_max, x.shape), dtype=x.dtype
            )
        return x
