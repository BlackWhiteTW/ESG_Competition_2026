import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import json
from datetime import datetime

from dataset import ESGDataset
from model import ESGMultiTaskModel


class ESGTrainer:
    """
    ESG 模型訓練器
    
    核心特性：
    1. 斷點續傳 (Checkpointing) - 解決 4 小時硬體限制
    2. 梯度累積 (Gradient Accumulation) - 模擬更大 Batch Size
    3. 自動混合精度 (AMP) - 節省 VRAM 並加快訓練
    4. 分層學習率 (Layered Learning Rate) - 微調預訓練模型的最佳實踐
    """
    
    def __init__(
        self,
        model: ESGMultiTaskModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 5,
        learning_rate: float = 2e-5,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.08,
        early_stopping_patience: int = 3,
        min_delta: float = 0.0,
        train_scope: str = "full-model",
        head_learning_rate_multiplier: float = 10.0,
        device: str = "cuda",
        checkpoint_dir: str = "models/checkpoints"
    ):
        """
        初始化訓練器
        
        Args:
            model: RoBERTa 多任務模型
            train_dataloader: 訓練資料加載器
            val_dataloader: 驗證資料加載器
            num_epochs: 總訓練 Epoch 數
            learning_rate: 初始學習率（適合 BERT 微調）
            gradient_accumulation_steps: 梯度累積步數（模擬大 Batch Size）
            warmup_steps: 線性學習率預熱步數（>0 時優先）
            warmup_ratio: 預熱步數比例（warmup_steps=0 時生效）
            early_stopping_patience: 早停耐心值
            min_delta: 視為改善的最小驗證損失下降幅度
            train_scope: 訓練範圍（full-model 或 heads-only）
            head_learning_rate_multiplier: 任務頭相對於 encoder 的學習率倍率
            device: 計算設備 (cuda/cpu)
            checkpoint_dir: 檢查點保存目錄
        """
        self.model = model.to(device)
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.train_scope = train_scope
        self.checkpoint_dir = checkpoint_dir
        
        # 建立檢查點目錄
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # ========== 優化器與學習率安排 ==========
        # 分層學習率：編碼器用較低 LR，任務頭用較高 LR
        encoder_params = list(self.model.encoder.parameters())
        head_params = [p for n, p in self.model.named_parameters() if 'encoder' not in n]

        if train_scope == "heads-only":
            for parameter in encoder_params:
                parameter.requires_grad = False
            optimizer_grouped_params = [
                {
                    'params': head_params,
                    'lr': learning_rate,
                    'weight_decay': 0.01
                }
            ]
        else:
            for parameter in encoder_params:
                parameter.requires_grad = True
            optimizer_grouped_params = [
                {
                    'params': encoder_params,
                    'lr': learning_rate,
                    'weight_decay': 0.01
                },
                {
                    'params': head_params,
                    'lr': learning_rate * head_learning_rate_multiplier,
                    'weight_decay': 0.01
                }
            ]
        
        self.optimizer = AdamW(optimizer_grouped_params)
        
        # 計算總訓練步數（含尾批次）
        steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        total_steps = max(1, steps_per_epoch * num_epochs)
        if warmup_steps > 0:
            effective_warmup_steps = min(warmup_steps, total_steps - 1) if total_steps > 1 else 0
        else:
            effective_warmup_steps = int(total_steps * warmup_ratio)
            effective_warmup_steps = min(effective_warmup_steps, total_steps - 1) if total_steps > 1 else 0

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=effective_warmup_steps,
            num_training_steps=total_steps
        )
        
        # ========== 自動混合精度 (AMP) ==========
        self.scaler = GradScaler()
        self.use_amp = device == "cuda"
        
        # ========== 訓練狀態追蹤 ==========
        self.best_val_loss = float('inf')
        self.bad_epochs = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch: int) -> float:
        """
        訓練單一 Epoch
        
        Returns:
            平均訓練損失
        """
        self.model.train()
        total_loss = 0.0
        accumulated_loss = 0.0
        
        # 使用 tqdm 建立進度條
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs}",
            total=len(self.train_dataloader)
        )

        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # 將批次移到設備
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # ========== 前向傳播 ==========
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids']
                )
                loss, loss_breakdown = self.model.compute_loss(outputs, batch)
                
                # 標準化損失（梯度累積）
                loss = loss / self.gradient_accumulation_steps
            
            # ========== 反向傳播 ==========
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_loss += loss.item()
            
            # ========== 梯度累積與優化器更新 ==========
            if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_dataloader):
                # 梯度裁剪（防止梯度爆炸）
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # 優化器步進
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                total_loss += accumulated_loss
                accumulated_loss = 0.0
            
            # 更新進度條
            progress_bar.set_postfix({
                'loss': loss.item() * self.gradient_accumulation_steps,
                'lr': self.scheduler.get_last_lr()[0]
            })
        
        num_optimizer_steps = math.ceil(len(self.train_dataloader) / self.gradient_accumulation_steps)
        avg_loss = total_loss / max(1, num_optimizer_steps)
        return avg_loss
    
    def validate(self) -> float:
        """
        驗證模型
        
        Returns:
            平均驗證損失
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids']
                    )
                    loss, _ = self.model.compute_loss(outputs, batch)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, step: int, is_best: bool = False):
        """
        保存訓練檢查點
        
        Args:
            epoch: 當前 Epoch
            step: 當前步驟
            is_best: 是否為最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss
        }
        
        # 保存到臨時位置
        temp_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{step}.pt")
        torch.save(checkpoint, temp_path)
        
        # 如果是最佳模型，額外保存一份
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"\n✅ 最佳模型已保存: {best_path}")
            print(f"   最佳驗證損失: {self.best_val_loss:.4f}")
        
        print(f"✅ 檢查點已保存: {temp_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, int]:
        """
        載入訓練檢查點
        
        Returns:
            (起始 Epoch, 起始步驟)
        """
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  檢查點不存在: {checkpoint_path}")
            return 0, 0
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.use_amp and checkpoint.get('scaler_state_dict') is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        })
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        next_epoch = epoch + 1
        
        print(f"✅ 檢查點已載入: {checkpoint_path}")
        print(f"🔄 已完成到 Epoch {epoch + 1}，將從 Epoch {next_epoch + 1} 繼續訓練")
        
        return next_epoch, step
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        完整訓練循環 (包含斷點續傳機制)
        
        Args:
            resume_from_checkpoint: 要恢復的檢查點路徑 (若 None，從頭開始)
        """
        start_epoch = 0
        
        # 嘗試載入檢查點
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            start_epoch, _ = self.load_checkpoint(resume_from_checkpoint)
        
        print("\n" + "=" * 80)
        print("🚀 開始訓練")
        print("=" * 80)
        print(f"設備: {self.device}")
        print(f"訓練範圍: {self.train_scope}")
        print(f"混合精度 (AMP): {self.use_amp}")
        print(f"梯度累積步數: {self.gradient_accumulation_steps}")
        print(f"訓練樣本: {len(self.train_dataloader) * self.train_dataloader.batch_size}")
        print(f"驗證樣本: {len(self.val_dataloader) * self.val_dataloader.batch_size}")
        print("=" * 80 + "\n")
        
        for epoch in range(start_epoch, self.num_epochs):
            # 訓練
            train_loss = self.train_epoch(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['learning_rate'].append(
                self.scheduler.get_last_lr()[0]
            )
            
            # 驗證
            val_loss = self.validate()
            self.training_history['val_loss'].append(val_loss)
            
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  訓練損失: {train_loss:.4f}")
            print(f"  驗證損失: {val_loss:.4f}")
            print(f"  學習率: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # 檢查是否為最佳模型
            is_best = val_loss < (self.best_val_loss - self.min_delta)
            if is_best:
                self.best_val_loss = val_loss
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            
            # 每個 Epoch 保存檢查點
            self.save_checkpoint(epoch, len(self.train_dataloader), is_best=is_best)

            if self.bad_epochs >= self.early_stopping_patience:
                print(f"\n⏹️  觸發 Early Stopping：連續 {self.bad_epochs} 個 epoch 無改善")
                break
        
        print("\n" + "=" * 80)
        print("✅ 訓練完成！")
        print(f"最佳驗證損失: {self.best_val_loss:.4f}")
        print("=" * 80)
        
        # 保存訓練歷史
        history_path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"\n訓練歷史已保存: {history_path}")


def create_data_splits(
    dataset: ESGDataset,
    train_ratio: float = 0.8,
    batch_size: int = 8,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    將資料集分割為訓練和驗證集
    
    Args:
        dataset: ESG 資料集
        train_ratio: 訓練集比例
        batch_size: 批次大小
        seed: 隨機種子（確保可重現性）
    
    Returns:
        (訓練加載器, 驗證加載器)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows 環境下設為 0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_dataloader, val_dataloader
