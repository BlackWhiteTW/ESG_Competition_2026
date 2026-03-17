import torch
import torch.nn as nn


class ESGDetectionHead(nn.Module):
    """
    處理 Yes/No 的二元判定 (Promise & Evidence status)。
    優化：使用更深層的 MLP 處理邏輯判定。
    """

    def __init__(self, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, cls_vector: torch.Tensor) -> torch.Tensor:
        return self.detector(cls_vector)
