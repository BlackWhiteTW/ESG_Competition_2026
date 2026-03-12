import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Tuple


class ESGMultiTaskModel(nn.Module):
    """
    ESG 永續承諾驗證多任務模型
    
    任務分解：
    1. Promise Detection: 二元分類 (是否有承諾)
    2. Promise Extraction: 序列標籤 (承諾的起始和結束位置)
    3. Evidence Detection: 二元分類 (是否有證據)
    4. Evidence Extraction: 序列標籤 (證據的起始和結束位置)
    5. ESG Classification: 3-way 分類 (E/S/G)
    6. Timeline Prediction: 4-way 分類 (已達成/2-5年/5年以上/未知)
    
    多任務學習的優勢：
    - 共享底層 RoBERTa 編碼層，參數更少
    - 不同任務互相參照，提升整體 F1-Score
    - 在 8GB VRAM 的環境下，訓練效率最高
    """
    
    def __init__(
        self,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        hidden_size: int = 768,
        num_labels_esg: int = 3,
        num_labels_timeline: int = 4,
        num_labels_quality: int = 3,
        dropout_rate: float = 0.1
    ):
        """
        初始化多任務模型
        
        Args:
            model_name: Hugging Face 預訓練模型
            hidden_size: 隱藏層維度 (BERT-base: 768, BERT-large: 1024)
            num_labels_esg: ESG 分類標籤數 (E/S/G)
            num_labels_timeline: 時間軸標籤數
            num_labels_quality: 證據品質標籤數 (Clear/Not Clear/N/A)
            dropout_rate: Dropout 比率（防止過擬合）
        """
        super().__init__()
        
        # 載入預訓練的編碼器
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size
        
        # ========== 共享層 ==========
        self.dropout = nn.Dropout(dropout_rate)
        
        # ========== 任務 1: Promise Detection (二元分類) ==========
        self.promise_detection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 2)  # 2 classes: Yes/No
        )
        
        # ========== 任務 2 & 3: Promise & Evidence Extraction (Token-level 分類) ==========
        # 改為標準的 QA 模式：對每個 Token 預測其為 start 或 end 的機率
        self.promise_span_head = nn.Linear(hidden_size, 2)  # 輸出: (start_logits, end_logits) per token
        self.evidence_span_head = nn.Linear(hidden_size, 2) # 輸出: (start_logits, end_logits) per token
        
        # ========== 任務 4: Evidence Detection (二元分類) ==========
        self.evidence_detection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 2)  # 2 classes: Yes/No
        )
        
        # ========== 任務 5: ESG Classification (3-way 分類) ==========
        self.esg_classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_labels_esg)  # 3 classes: E/S/G
        )
        
        # ========== 任務 6: Timeline Prediction (4-way 分類) ==========
        self.timeline_classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_labels_timeline)  # 4 classes
        )

        # ========== 任務 7: Evidence Quality Prediction (3-way 分類) ==========
        self.quality_classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_labels_quality)  # 3 classes: Clear/Not Clear/N/A
        )
        
        # ========== 損失函數定義 ==========
        self.ce_loss = nn.CrossEntropyLoss()  # 用於分類任務
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: 注意力遮罩 (batch_size, seq_length)
            token_type_ids: Token 類型 IDs (batch_size, seq_length)
        
        Returns:
            包含所有任務的輸出字典
        """
        # ========== 編碼層 ==========
        # 使用預訓練的 RoBERTa 編碼器
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False
        )
        
        # last_hidden_state: (batch_size, seq_length, hidden_size)
        sequence_output = encoder_output.last_hidden_state
        
        # 用 [CLS] Token 的表示進行文檔級分類
        cls_output = sequence_output[:, 0, :]  # (batch_size, hidden_size)
        
        # ========== 多任務輸出 ==========
        outputs = {}
        
        # 任務 1: Promise Detection (文檔級)
        outputs['promise_logits'] = self.promise_detection_head(self.dropout(cls_output))
        
        # 任務 2: Promise Span Extraction (Token-level)
        # promise_span_logits: (batch_size, seq_length, 2)
        promise_span_logits = self.promise_span_head(sequence_output)
        outputs['promise_start_logits'] = promise_span_logits[:, :, 0]
        outputs['promise_end_logits'] = promise_span_logits[:, :, 1]
        
        # 任務 3: Evidence Detection (文檔級)
        outputs['evidence_logits'] = self.evidence_detection_head(self.dropout(cls_output))
        
        # 任務 4: Evidence Span Extraction (Token-level)
        # evidence_span_logits: (batch_size, seq_length, 2)
        evidence_span_logits = self.evidence_span_head(sequence_output)
        outputs['evidence_start_logits'] = evidence_span_logits[:, :, 0]
        outputs['evidence_end_logits'] = evidence_span_logits[:, :, 1]
        
        # 任務 5: ESG Classification (文檔級)
        outputs['esg_logits'] = self.esg_classification_head(self.dropout(cls_output))
        
        # 任務 6: Timeline Classification (文檔級)
        outputs['timeline_logits'] = self.timeline_classification_head(self.dropout(cls_output))

        # 任務 7: Evidence Quality Classification (文檔級)
        outputs['quality_logits'] = self.quality_classification_head(self.dropout(cls_output))
        
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        task_weights: Dict[str, float] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        計算多任務加權損失
        """
        if task_weights is None:
            task_weights = {
                'promise_loss': 1.0,
                'promise_span_loss': 2.0,  # 提升 span 權重
                'evidence_loss': 1.0,
                'evidence_span_loss': 2.0,
                'esg_loss': 1.0,
                'timeline_loss': 1.0,
                'quality_loss': 1.0
            }
        
        losses = {}
        total_loss = torch.tensor(0.0, device=outputs['promise_logits'].device)
        
        # 1. Promise Detection Loss
        if 'promise_status' in batch:
            promise_loss = self.ce_loss(outputs['promise_logits'], batch['promise_status'])
            losses['promise_loss'] = promise_loss.item()
            total_loss += task_weights['promise_loss'] * promise_loss
        
        # 2. Promise Span Loss (Token-level CrossEntropy)
        if 'promise_start' in batch and 'promise_end' in batch:
            # 只在 promise_status 為 Yes 的樣本上計算 Span Loss 以避免噪音
            mask = (batch['promise_status'] == 1)
            if mask.any():
                start_loss = self.ce_loss(outputs['promise_start_logits'][mask], batch['promise_start'][mask])
                end_loss = self.ce_loss(outputs['promise_end_logits'][mask], batch['promise_end'][mask])
                span_loss = (start_loss + end_loss) / 2
                losses['promise_span_loss'] = span_loss.item()
                total_loss += task_weights['promise_span_loss'] * span_loss
            else:
                losses['promise_span_loss'] = 0.0
        
        # 3. Evidence Detection Loss
        if 'evidence_status' in batch:
            evidence_loss = self.ce_loss(outputs['evidence_logits'], batch['evidence_status'])
            losses['evidence_loss'] = evidence_loss.item()
            total_loss += task_weights['evidence_loss'] * evidence_loss
        
        # 4. Evidence Span Loss
        if 'evidence_start' in batch and 'evidence_end' in batch:
            mask = (batch['evidence_status'] == 1)
            if mask.any():
                start_loss = self.ce_loss(outputs['evidence_start_logits'][mask], batch['evidence_start'][mask])
                end_loss = self.ce_loss(outputs['evidence_end_logits'][mask], batch['evidence_end'][mask])
                span_loss = (start_loss + end_loss) / 2
                losses['evidence_span_loss'] = span_loss.item()
                total_loss += task_weights['evidence_span_loss'] * span_loss
            else:
                losses['evidence_span_loss'] = 0.0
        
        # 5. ESG Loss
        if 'esg_label' in batch:
            esg_loss = self.ce_loss(outputs['esg_logits'], batch['esg_label'])
            losses['esg_loss'] = esg_loss.item()
            total_loss += task_weights['esg_loss'] * esg_loss
        
        # 6. Timeline Loss
        if 'timeline_label' in batch:
            timeline_loss = self.ce_loss(outputs['timeline_logits'], batch['timeline_label'])
            losses['timeline_loss'] = timeline_loss.item()
            total_loss += task_weights['timeline_loss'] * timeline_loss

        # 7. Quality Loss
        if 'quality_label' in batch:
            quality_loss = self.ce_loss(outputs['quality_logits'], batch['quality_label'])
            losses['quality_loss'] = quality_loss.item()
            total_loss += task_weights['quality_loss'] * quality_loss
        
        losses['total_loss'] = total_loss.item()
        return total_loss, losses


if __name__ == "__main__":
    # ========== 簡單的模型測試 ==========
    print("🧪 ESG 多任務模型測試")
    print("=" * 80)
    
    # 建立模型
    model = ESGMultiTaskModel(
        model_name="hfl/chinese-roberta-wwm-ext",
        dropout_rate=0.1
    )
    
    print(f"✅ 模型已載入")
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 建立假的輸入
    batch_size = 4
    seq_length = 128
    
    dummy_input = {
        'input_ids': torch.randint(0, 21128, (batch_size, seq_length)),
        'attention_mask': torch.ones(batch_size, seq_length),
        'token_type_ids': torch.zeros(batch_size, seq_length, dtype=torch.long),
        'promise_status': torch.randint(0, 2, (batch_size,)),
        'promise_start': torch.randint(0, seq_length, (batch_size,)),
        'promise_end': torch.randint(0, seq_length, (batch_size,)),
        'evidence_status': torch.randint(0, 2, (batch_size,)),
        'evidence_start': torch.randint(0, seq_length, (batch_size,)),
        'evidence_end': torch.randint(0, seq_length, (batch_size,)),
        'esg_label': torch.randint(0, 3, (batch_size,)),
        'timeline_label': torch.randint(0, 4, (batch_size,)),
    }
    
    # 前向傳播
    outputs = model(
        input_ids=dummy_input['input_ids'],
        attention_mask=dummy_input['attention_mask'],
        token_type_ids=dummy_input['token_type_ids'],
    )
    
    print(f"\n✅ 前向傳播成功")
    print(f"輸出形狀:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
    
    # 計算損失
    loss, loss_breakdown = model.compute_loss(outputs, dummy_input)
    
    print(f"\n✅ 損失計算成功")
    print(f"損失詳細:")
    for task, loss_val in loss_breakdown.items():
        print(f"  {task}: {loss_val:.4f}")
