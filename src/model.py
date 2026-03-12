import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Tuple

# 導入拆分後的任務頭模組
from tasks.detection import ESGDetectionHead
from tasks.extraction import ESGExtractionHead
from tasks.classification import ESGClassificationHead

class ESGMultiTaskModel(nn.Module):
    """
    ESG 永續承諾驗證多任務模型 (模組化版本)
    
    將複雜的任務頭拆分為獨立模組，方便針對特定任務進行優化 (例如 BiLSTM 擷取器)。
    """
    
    def __init__(
        self,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        hidden_size: int = 768,
        num_labels_esg: int = 3,
        num_labels_timeline: int = 5,
        num_labels_quality: int = 4,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # 載入核心編碼器
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        
        # ========== 模組化任務頭組合 ==========
        
        # 1. 承諾相關 (Promise)
        self.promise_detection = ESGDetectionHead(hidden_size, dropout_rate)
        self.promise_extraction = ESGExtractionHead(hidden_size, dropout_rate)
        
        # 2. 證據相關 (Evidence)
        self.evidence_detection = ESGDetectionHead(hidden_size, dropout_rate)
        self.evidence_extraction = ESGExtractionHead(hidden_size, dropout_rate)
        
        # 3. 分類相關 (ESG, Timeline, Quality)
        self.esg_classifier = ESGClassificationHead(hidden_size, num_labels_esg, dropout_rate)
        self.timeline_classifier = ESGClassificationHead(hidden_size, num_labels_timeline, dropout_rate)
        self.quality_classifier = ESGClassificationHead(hidden_size, num_labels_quality, dropout_rate)
        
        # ========== 損失函數 ==========
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        
        # 1. 通過編碼器
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = encoder_output.last_hidden_state
        cls_output = sequence_output[:, 0, :]  # 取 [CLS] 向量
        
        # 2. 呼叫各模組任務頭
        outputs = {}
        
        # 承諾任務
        outputs['promise_logits'] = self.promise_detection(self.dropout(cls_output))
        p_start, p_end = self.promise_extraction(sequence_output)
        outputs['promise_start_logits'] = p_start
        outputs['promise_end_logits'] = p_end
        
        # 證據任務
        outputs['evidence_logits'] = self.evidence_detection(self.dropout(cls_output))
        e_start, e_end = self.evidence_extraction(sequence_output)
        outputs['evidence_start_logits'] = e_start
        outputs['evidence_end_logits'] = e_end
        
        # 分類任務
        outputs['esg_logits'] = self.esg_classifier(self.dropout(cls_output))
        outputs['timeline_logits'] = self.timeline_classifier(self.dropout(cls_output))
        outputs['quality_logits'] = self.quality_classifier(self.dropout(cls_output))
        
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        task_weights: Dict[str, float] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        加權損失計算 (保持與前一版優化邏輯一致)
        """
        if task_weights is None:
            task_weights = {
                'promise_loss': 1.0,
                'promise_span_loss': 5.0,
                'evidence_loss': 1.0,
                'evidence_span_loss': 5.0,
                'esg_loss': 2.0,
                'timeline_loss': 1.0,
                'quality_loss': 1.0
            }
        
        losses = {}
        total_loss = torch.tensor(0.0, device=outputs['promise_logits'].device)
        
        # 損失計算邏輯 (與優化版相同，確保收斂穩定)
        # 1. Promise Detection
        if 'promise_status' in batch:
            loss = self.ce_loss(outputs['promise_logits'], batch['promise_status'])
            losses['promise_loss'] = loss.item()
            total_loss += task_weights['promise_loss'] * loss
        
        # 2. Promise Span
        if 'promise_start' in batch and 'promise_end' in batch:
            mask = (batch['promise_status'] == 1)
            if mask.any():
                span_loss = (
                    self.ce_loss(outputs['promise_start_logits'][mask], batch['promise_start'][mask]) +
                    self.ce_loss(outputs['promise_end_logits'][mask], batch['promise_end'][mask])
                ) / 2
                losses['promise_span_loss'] = span_loss.item()
                total_loss += task_weights['promise_span_loss'] * span_loss
        
        # 3. Evidence Detection
        if 'evidence_status' in batch:
            loss = self.ce_loss(outputs['evidence_logits'], batch['evidence_status'])
            losses['evidence_loss'] = loss.item()
            total_loss += task_weights['evidence_loss'] * loss
            
        # 4. Evidence Span
        if 'evidence_start' in batch and 'evidence_end' in batch:
            mask = (batch['evidence_status'] == 1)
            if mask.any():
                span_loss = (
                    self.ce_loss(outputs['evidence_start_logits'][mask], batch['evidence_start'][mask]) +
                    self.ce_loss(outputs['evidence_end_logits'][mask], batch['evidence_end'][mask])
                ) / 2
                losses['evidence_span_loss'] = span_loss.item()
                total_loss += task_weights['evidence_span_loss'] * span_loss
                
        # 5. ESG 分類
        if 'esg_label' in batch:
            loss = self.ce_loss(outputs['esg_logits'], batch['esg_label'])
            losses['esg_loss'] = loss.item()
            total_loss += task_weights['esg_loss'] * loss
            
        # 6. Timeline 分類 (有承諾才算)
        if 'timeline_label' in batch:
            mask = (batch['promise_status'] == 1)
            if mask.any():
                loss = self.ce_loss(outputs['timeline_logits'][mask], batch['timeline_label'][mask])
                losses['timeline_loss'] = loss.item()
                total_loss += task_weights['timeline_loss'] * loss

        # 7. Quality 分類 (有證據才算)
        if 'quality_label' in batch:
            mask = (batch['evidence_status'] == 1)
            if mask.any():
                loss = self.ce_loss(outputs['quality_logits'][mask], batch['quality_label'][mask])
                losses['quality_loss'] = loss.item()
                total_loss += task_weights['quality_loss'] * loss
        
        losses['total_loss'] = total_loss.item()
        return total_loss, losses
