import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import json
from datetime import datetime
from sklearn.model_selection import KFold

from dataset import ESGDataset
from model import ESGMultiTaskModel
from analyzer import ESGMockEvaluator
from inference import ESGInference


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
        checkpoint_dir: str = "models/checkpoints"
    ):
        self.model = model.to(device)
        self.device = device
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
                "params": [p for n, p in self.model.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
                "lr": learning_rate * 0.1, # Encoder LR = 1/10
            },
            {
                "params": [p for n, p in self.model.encoder.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": learning_rate * 0.1,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if "encoder" not in n],
                "weight_decay": 0.01,
                "lr": learning_rate, # Heads LR = Full
            },
        ]
        
        self.optimizer = AdamW(optimizer_grouped_params)
        
        steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        total_steps = steps_per_epoch * num_epochs
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps
        )
        
        self.scaler = GradScaler()
        self.use_amp = device == "cuda"
        self.best_score = -1.0  # 改用 Score 作為儲存基準
        self.bad_epochs = 0
        
        # 初始化模擬評估器
        inf_engine = ESGInference(self.model, [], device=device)
        self.evaluator = ESGMockEvaluator(inf_engine)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with autocast(enabled=self.use_amp):
                outputs = self.model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])
                loss, _ = self.model.compute_loss(outputs, batch)
                loss = loss / self.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                total_loss += loss.item() * self.gradient_accumulation_steps
            
            progress_bar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
        return total_loss / len(self.train_dataloader)

    def validate_score(self):
        """核心：根據競賽指標評估"""
        self.model.eval()
        # 由於訓練中 Evaluator 共享 model 物件，直接評估
        report = self.evaluator.analyze_performance(self.val_dataloader.dataset, silent=True)
        return report['TOTAL_SCORE']

    def train(self, resume_from_checkpoint=None):
        print(f"[INFO] 開始訓練，基準指標: TOTAL_SCORE")
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            current_score = self.validate_score()
            
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, TOTAL_SCORE={current_score:.4f}")
            
            if current_score > self.best_score:
                self.best_score = current_score
                self.bad_epochs = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.bad_epochs += 1
            
            if self.bad_epochs >= self.early_stopping_patience:
                print(f"[STOP] Early Stopping 觸發")
                break

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_score': self.best_score
        }
        if is_best:
            path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, path)
            print(f"[SUCCESS] 發現更優模型！Score: {self.best_score:.4f} 已保存")

def create_data_splits(dataset, train_ratio=0.8, batch_size=8, seed=42):
    torch.manual_seed(seed)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
           DataLoader(val_ds, batch_size=batch_size, shuffle=False)

def create_kfold_splits(dataset, num_folds=5, fold_idx=0, batch_size=8, seed=42):
    indices = list(range(len(dataset)))
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = list(kf.split(indices))
    train_indices, val_indices = folds[fold_idx]
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
           DataLoader(val_ds, batch_size=batch_size, shuffle=False)
