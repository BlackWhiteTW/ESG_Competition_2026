import torch
import torch.nn as nn

class ESGExtractionHead(nn.Module):
    """
    處理語句擷取 (Span Extraction)。
    優化：加入 BiLSTM 層，捕捉 Token 之間的序列資訊，解決「只能擷取一小段」的問題。
    """
    def __init__(self, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        # 加入 BiLSTM 強化 Token 之間的上下文連結
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if dropout_rate > 0 else 0
        )
        
        # 預測每個 Token 是 Start 或 End 的機率
        self.span_classifier = nn.Linear(hidden_size, 2)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # sequence_output: (batch_size, seq_len, hidden_size)
        
        # 1. 通過 BiLSTM 增強序列理解
        lstm_out, _ = self.lstm(sequence_output)
        
        # 2. 產出 Start/End Logits (batch_size, seq_len, 2)
        logits = self.span_classifier(lstm_out)
        
        # 切分為 start 和 end (batch_size, seq_len)
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]
        
        return start_logits, end_logits
