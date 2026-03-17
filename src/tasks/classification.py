import torch
import torch.nn as nn


class ESGClassificationHead(nn.Module):
    """
    處理 ESG 類別、時間軸與品質的分類器。
    優化：加入中間投影層提升分類深度。
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, cls_vector: torch.Tensor) -> torch.Tensor:
        return self.classifier(cls_vector)
