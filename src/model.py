import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import config

# 導入拆分後的任務頭模組
from tasks.detection import ESGDetectionHead
from tasks.extraction import ESGExtractionHead
from tasks.classification import ESGClassificationHead


class FocalLoss(nn.Module):
    """
    競賽級 Focal Loss：解決樣本不平衡與假陽性問題
    """

    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class ESGMultiTaskModel(nn.Module):
    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        hidden_size: int = 1024,
        num_labels_esg: int = 3,
        num_labels_timeline: int = 5,
        num_labels_quality: int = 4,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        print(f"[INFO] 正在載入核心編碼器: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)

        # 強制維度對齊
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)

        # 任務頭
        self.promise_detection = ESGDetectionHead(hidden_size, dropout_rate)
        self.promise_extraction = ESGExtractionHead(hidden_size, dropout_rate)
        self.evidence_detection = ESGDetectionHead(hidden_size, dropout_rate)
        self.evidence_extraction = ESGExtractionHead(hidden_size, dropout_rate)
        self.esg_classifier = ESGClassificationHead(
            hidden_size, num_labels_esg, dropout_rate
        )
        self.timeline_classifier = ESGClassificationHead(
            hidden_size, num_labels_timeline, dropout_rate
        )
        self.quality_classifier = ESGClassificationHead(
            hidden_size, num_labels_quality, dropout_rate
        )

        # 損失函數：偵測任務用 Focal Loss，其餘用 Label Smoothing CE
        self.focal_loss = FocalLoss(gamma=2.0)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = encoder_output.last_hidden_state
        cls_output = sequence_output[:, 0, :]

        return {
            "promise_logits": self.promise_detection(self.dropout(cls_output)),
            "promise_bio_logits": self.promise_extraction(sequence_output),
            "evidence_logits": self.evidence_detection(self.dropout(cls_output)),
            "evidence_bio_logits": self.evidence_extraction(sequence_output),
            "esg_logits": self.esg_classifier(self.dropout(cls_output)),
            "timeline_logits": self.timeline_classifier(self.dropout(cls_output)),
            "quality_logits": self.quality_classifier(self.dropout(cls_output)),
        }

    def compute_loss(self, outputs, batch, task_weights=None):
        if task_weights is None:
            # 究極比例：強化證據偵測與時程判定
            task_weights = {
                "promise_loss": 15.0,  # 保持最高權重對抗連鎖懲罰
                "promise_bio_loss": 1.0,
                "evidence_loss": 12.0,  # 大幅提升 (8.0 -> 12.0)
                "evidence_bio_loss": 1.0,
                "esg_loss": 2.0,
                "timeline_loss": 5.0,  # 提升 (3.0 -> 5.0) 救回最弱項
                "quality_loss": 1.0,
            }

        losses = {}
        total_loss = torch.tensor(0.0, device=outputs["promise_logits"].device)

        # 使用 Focal Loss 處理關鍵判定
        l1 = self.focal_loss(outputs["promise_logits"], batch["promise_status"])
        l3 = self.focal_loss(outputs["evidence_logits"], batch["evidence_status"])

        total_loss += task_weights["promise_loss"] * l1
        total_loss += task_weights["evidence_loss"] * l3

        # 其餘任務使用 CE Loss
        total_loss += task_weights["promise_bio_loss"] * self.ce_loss(
            outputs["promise_bio_logits"].view(-1, 3), batch["promise_bio"].view(-1)
        )
        total_loss += task_weights["evidence_bio_loss"] * self.ce_loss(
            outputs["evidence_bio_logits"].view(-1, 3), batch["evidence_bio"].view(-1)
        )
        total_loss += task_weights["esg_loss"] * self.ce_loss(
            outputs["esg_logits"], batch["esg_label"]
        )

        mask_p = batch["promise_status"] == 1
        if mask_p.any():
            total_loss += task_weights["timeline_loss"] * self.ce_loss(
                outputs["timeline_logits"][mask_p], batch["timeline_label"][mask_p]
            )

        mask_e = batch["evidence_status"] == 1
        if mask_e.any():
            total_loss += task_weights["quality_loss"] * self.ce_loss(
                outputs["quality_logits"][mask_e], batch["quality_label"][mask_e]
            )

        losses["total_loss"] = total_loss.item()
        return total_loss, losses
