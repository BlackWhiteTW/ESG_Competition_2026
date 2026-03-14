import torch
import torch.nn as nn

class ESGExtractionHead(nn.Module):
    """
    處理語句擷取 (BIO Sequence Tagging)。
    優化：使用 3 分類器 (0: O, 1: B, 2: I) 支援非連續語句擷取。
    """
    def __init__(self, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 輸出 3 類: O, B, I
        self.bio_classifier = nn.Linear(hidden_size, 3)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # sequence_output: (batch_size, seq_len, hidden_size)
        
        # 1. 通過 BiLSTM
        lstm_out, _ = self.lstm(sequence_output)
        
        # 2. 產出 BIO Logits (batch_size, seq_len, 3)
        logits = self.bio_classifier(lstm_out)
        
        return logits
