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
    ESG 永續承諾驗證多任務模型 (BIO 序列標記版本 - 強制 Large 版)
    """
    
    def __init__(
        self,
        model_name: str = "hfl/chinese-roberta-wwm-ext-large",
        hidden_size: int = 1024,
        num_labels_esg: int = 3,
        num_labels_timeline: int = 5,
        num_labels_quality: int = 4,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        print(f"[INFO] 正在載入核心編碼器: {model_name} (維度: {hidden_size})")
        
        # 載入核心編碼器
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 強制檢查維度對齊
        encoder_hidden_size = self.encoder.config.hidden_size
        if encoder_hidden_size != hidden_size:
            print(f"[WARNING] 偵測到維度不匹配！修正 hidden_size 為 {encoder_hidden_size}")
            hidden_size = encoder_hidden_size

        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        
        # ========== 模組化任務頭組合 ==========
        self.promise_detection = ESGDetectionHead(hidden_size, dropout_rate)
        self.promise_extraction = ESGExtractionHead(hidden_size, dropout_rate)
        self.evidence_detection = ESGDetectionHead(hidden_size, dropout_rate)
        self.evidence_extraction = ESGExtractionHead(hidden_size, dropout_rate)
        self.esg_classifier = ESGClassificationHead(hidden_size, num_labels_esg, dropout_rate)
        self.timeline_classifier = ESGClassificationHead(hidden_size, num_labels_timeline, dropout_rate)
        self.quality_classifier = ESGClassificationHead(hidden_size, num_labels_quality, dropout_rate)
        
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = encoder_output.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        
        outputs = {}
        outputs['promise_logits'] = self.promise_detection(self.dropout(cls_output))
        outputs['promise_bio_logits'] = self.promise_extraction(sequence_output)
        outputs['evidence_logits'] = self.evidence_detection(self.dropout(cls_output))
        outputs['evidence_bio_logits'] = self.evidence_extraction(sequence_output)
        outputs['esg_logits'] = self.esg_classifier(self.dropout(cls_output))
        outputs['timeline_logits'] = self.timeline_classifier(self.dropout(cls_output))
        outputs['quality_logits'] = self.quality_classifier(self.dropout(cls_output))
        return outputs
    
    def compute_loss(self, outputs, batch, task_weights=None):
        if task_weights is None:
            task_weights = {
                'promise_loss': 1.0, 'promise_bio_loss': 5.0,
                'evidence_loss': 1.0, 'evidence_bio_loss': 5.0,
                'esg_loss': 2.0, 'timeline_loss': 1.0, 'quality_loss': 1.0
            }
        
        losses = {}
        total_loss = torch.tensor(0.0, device=outputs['promise_logits'].device)
        
        # 1. Promise Detection
        loss = self.ce_loss(outputs['promise_logits'], batch['promise_status'])
        losses['promise_loss'] = loss.item()
        total_loss += task_weights['promise_loss'] * loss
        
        # 2. Promise BIO
        logits = outputs['promise_bio_logits'].view(-1, 3)
        targets = batch['promise_bio'].view(-1)
        loss = self.ce_loss(logits, targets)
        losses['promise_bio_loss'] = loss.item()
        total_loss += task_weights['promise_bio_loss'] * loss
        
        # 3. Evidence Detection
        loss = self.ce_loss(outputs['evidence_logits'], batch['evidence_status'])
        losses['evidence_loss'] = loss.item()
        total_loss += task_weights['evidence_loss'] * loss
            
        # 4. Evidence BIO
        logits = outputs['evidence_bio_logits'].view(-1, 3)
        targets = batch['evidence_bio'].view(-1)
        loss = self.ce_loss(logits, targets)
        losses['evidence_bio_loss'] = loss.item()
        total_loss += task_weights['evidence_bio_loss'] * loss
                
        # 5. ESG 分類
        loss = self.ce_loss(outputs['esg_logits'], batch['esg_label'])
        losses['esg_loss'] = loss.item()
        total_loss += task_weights['esg_loss'] * loss
            
        # 6. Timeline 分類 (有承諾才算)
        mask = (batch['promise_status'] == 1)
        if mask.any():
            loss = self.ce_loss(outputs['timeline_logits'][mask], batch['timeline_label'][mask])
            losses['timeline_loss'] = loss.item()
            total_loss += task_weights['timeline_loss'] * loss

        # 7. Quality 分類 (有證據才算)
        mask = (batch['evidence_status'] == 1)
        if mask.any():
            loss = self.ce_loss(outputs['quality_logits'][mask], batch['quality_label'][mask])
            losses['quality_loss'] = loss.item()
            total_loss += task_weights['quality_loss'] * loss
        
        losses['total_loss'] = total_loss.item()
        return total_loss, losses
