import torch
from torch import nn

from .linear import LinearBlock


class GruBlock(nn.Module):
    def __init__(self, keypoints_len: int, hidden_size=128, only_last=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.concat_size = keypoints_len + self.hidden_size

        self.reset_gate = nn.Linear(self.concat_size, self.hidden_size)
        self.update_gate = nn.Linear(self.concat_size, self.hidden_size)
        self.weights = nn.Linear(self.concat_size, self.hidden_size)

        self.only_last = only_last

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        hidden_state = torch.Tensor(len(x), self.hidden_size)
        output = []
        for i in range(x.shape[1]):
            concat_x = torch.concat([hidden_state, x[:, i, :]], dim=1)
            reset_x = torch.sigmoid(self.reset_gate(concat_x))
            update_x = torch.sigmoid(self.update_gate(concat_x))
            hidden_state_update = torch.tanh(
                self.weights(torch.concat([reset_x * hidden_state, x[:, i, :]], dim=1))
            )
            hidden_state = (1 - update_x) * hidden_state + update_x * hidden_state_update
            output.append(hidden_state.clone())

        if self.only_last:
            return output[-1]
        return torch.stack(output, dim=1)


class GruModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        keypoints_len: int,
        dim_no: int = 3,
    ):

        flattened_size = keypoints_len * dim_no
        super().__init__()
        self.gru_model = nn.Sequential(
            nn.Flatten(start_dim=2),
            torch.nn.BatchNorm1d(100),
            nn.GRU(flattened_size, 128, num_layers=2, batch_first=True, bidirectional=True),
            # GruBlock(flattened_size, 128),
            # GruBlock(128, 64, only_last=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            LinearBlock(512, 256, batch_norm=True),
            LinearBlock(256, num_classes, activation=None),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru_model(x)
        h_n = h_n.permute(1, 0, 2)
        return self.head(h_n)
