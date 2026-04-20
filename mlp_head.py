import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    """Projection head used after flattening the HybridCnnDinoNet feature map."""

    def __init__(self, in_dim, hidden_dim=2048, hidden_dim2=1024, out_dim=512):
        super().__init__()
        if hidden_dim2 is not None:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim2),
                nn.BatchNorm1d(hidden_dim2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim2, out_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x):
        x = self.mlp(x)
        return F.normalize(x, dim=1)
