import os

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import torch
import glob
from datetime import datetime
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.model_selection import KFold
from model import ESGMultiTaskModel
from analyzer import ESGMockEvaluator
from inference import ESGInference


import config


class ESGTrainer:
    def __init__(
        self,
        model: ESGMultiTaskModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 5,
        learning_rate: float = 2e-5,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.1,
        early_stopping_patience: int = 5,
        train_scope: str = "full-model",
        device: str = "cuda",
        checkpoint_dir: str = config.MODELS_DIR,
        fold_idx: int = 0,
    ):
        self.model = model.to(device)
        self.device = device
        self.fold_idx = fold_idx
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        # 差異化學習率設定
        # Encoder 用極小學習率保護知識，Heads 用較大學習率快速收斂
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_params = [
            {
                "params": [
                    p
                    for n, p in self.model.encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
                "lr": learning_rate * 0.1,
            },
            {
                "params": [
                    p
                    for n, p in self.model.encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": learning_rate * 0.1,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if "encoder" not in n
                ],
                "weight_decay": 0.01,
                "lr": learning_rate,
            },
        ]

        self.optimizer = AdamW(optimizer_grouped_params)

        steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        total_steps = steps_per_epoch * num_epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )

        self.scaler = GradScaler()
        self.use_amp = device == "cuda"
        self.best_score = -1.0
        self.bad_epochs = 0

        inf_engine = ESGInference(self.model, [], device=device)
        self.evaluator = ESGMockEvaluator(inf_engine)

    def _log(self, message):
        """帶時間戳記的日誌輸出"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with autocast(
                device_type=("cuda" if "cuda" in self.device else "cpu"),
                enabled=self.use_amp,
            ):
                outputs = self.model(
                    batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
                )
                loss, _ = self.model.compute_loss(outputs, batch)
                loss = loss / self.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                total_loss += loss.item() * self.gradient_accumulation_steps

            progress_bar.set_postfix(
                {"loss": loss.item() * self.gradient_accumulation_steps}
            )
        return total_loss / len(self.train_dataloader)

    def validate_score(self):
        """核心：根據競賽指標評估"""
        self.model.eval()
        report = self.evaluator.analyze_performance(
            self.val_dataloader.dataset, silent=True
        )
        return report["TOTAL_SCORE"]

    def train(self, resume_from_checkpoint=None):
        start_epoch = 0
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            start_epoch = self.load_checkpoint(resume_from_checkpoint)
            self._log(f"從 Epoch {start_epoch} 恢復訓練")

        self._log("開始訓練流程，基準指標: TOTAL_SCORE")
        for epoch in range(start_epoch, self.num_epochs):
            train_loss = self.train_epoch(epoch)
            current_score = self.validate_score()

            self._log(
                f"Epoch {epoch + 1} 完成: Loss={train_loss:.4f}, TOTAL_SCORE={current_score:.4f}"
            )

            # 每個 Epoch 結束都儲存進度檢查點
            self.save_checkpoint(epoch, is_best=False)

            if current_score > self.best_score:
                self.best_score = current_score
                self.bad_epochs = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.bad_epochs += 1

            if self.bad_epochs >= self.early_stopping_patience:
                self._log("Early Stopping 觸發，停止訓練。")
                break

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_score = checkpoint.get("best_score", -1.0)
        # 返回下一個起始 Epoch
        return checkpoint.get("epoch", 0) + 1

    def save_checkpoint(self, epoch, is_best=False):
        # 準備完整的狀態字典以供恢復
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_score": self.best_score,
            "fold_idx": self.fold_idx,
        }

        # 建立該 Fold 專屬資料夾
        fold_dir = os.path.join(self.checkpoint_dir, f"kfold_{self.fold_idx + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        if is_best:
            # 最佳模型依舊放在根目錄，方便 merge_models.py 抓取
            filename = f"best_model_{self.fold_idx + 1}.pt"
            path = os.path.join(self.checkpoint_dir, filename)
            torch.save(checkpoint, path)

            # 同時建立一個不帶編號的複本
            base_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, base_path)

            self._log(
                f"發現更優模型！已存至 {path} (Score: {self.best_score:.4f})"
            )
        else:
            # [優化] 進度檢查點：改為每 5 輪儲存一次，且存放在子資料夾
            if (epoch + 1) % 5 == 0:
                filename = f"checkpoint_kfold_{self.fold_idx + 1}_epoch_{epoch + 1}.pt"
                path = os.path.join(fold_dir, filename)
                
                # 儲存前先嘗試清理舊的進度檢查點
                # 規則：不刪除 best_model, 不刪除第 3 輪與第 6 輪的進度點
                old_checkpoints = glob.glob(os.path.join(fold_dir, "checkpoint_kfold_*.pt"))
                for old_ckpt in old_checkpoints:
                    # 檢查檔名，如果是 epoch_3 或 epoch_6 則保留
                    if "_epoch_3.pt" in old_ckpt or "_epoch_6.pt" in old_ckpt:
                        continue
                    try: 
                        os.remove(old_ckpt)
                    except: 
                        pass

                torch.save(checkpoint, path)
                self._log(f"已儲存第 {epoch + 1} 輪進度點: {path}")


def create_data_splits(dataset, train_ratio=0.8, batch_size=8, seed=42):
    torch.manual_seed(seed)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )


def create_kfold_splits(dataset, num_folds=5, fold_idx=0, batch_size=8, seed=42):
    indices = list(range(len(dataset)))
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = list(kf.split(indices))
    train_indices, val_indices = folds[fold_idx]
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )
