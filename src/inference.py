import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from dataset import ESGDataset
from model import ESGMultiTaskModel


class ESGInference:
    """
    ESG 模型推理器
    
    功能：
    1. 載入訓練好的模型
    2. 對測試資料進行預測
    3. 將模型輸出轉換為可提交的格式
    4. 邊界情況處理 (post-processing)
    """
    
    def __init__(
        self,
        model: ESGMultiTaskModel,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化推理器
        
        Args:
            model: RoBERTa 多任務模型
            checkpoint_path: 訓練好的模型檢查點
            device: 計算設備
        """
        self.model = model.to(device)
        self.device = device
        
        # 載入檢查點
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 模型已載入: {checkpoint_path}")
        else:
            print(f"⚠️  警告: 檢查點不存在 {checkpoint_path}")
        
        self.model.eval()
        
        # Token 到標籤的映射
        self.esg_labels = {0: 'E', 1: 'S', 2: 'G'}
        self.timeline_labels = {
            0: 'already',
            1: 'within_2_years',
            2: 'between_2_and_5_years',
            3: 'more_than_5_years',
            4: 'N/A'
        }
        self.quality_labels = {
            0: 'Clear',
            1: 'Not Clear',
            2: 'Misleading',
            3: 'N/A'
        }
    
    def predict_batch(
        self,
        batch: Dict[str, torch.Tensor],
        promise_threshold: float = 0.5,
        evidence_threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        批次預測
        """
        batch_input = {
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
            'token_type_ids': batch['token_type_ids'].to(self.device)
        }
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch_input['input_ids'],
                attention_mask=batch_input['attention_mask'],
                token_type_ids=batch_input['token_type_ids']
            )
        
        predictions = {}
        
        # Promise Detection
        promise_probs = torch.softmax(outputs['promise_logits'], dim=1)
        predictions['promise_status'] = (promise_probs[:, 1] > promise_threshold).int()
        
        # Promise Extraction (Token-level argmax)
        predictions['promise_start'] = outputs['promise_start_logits'].argmax(dim=1)
        predictions['promise_end'] = outputs['promise_end_logits'].argmax(dim=1)
        
        # Evidence Detection
        evidence_probs = torch.softmax(outputs['evidence_logits'], dim=1)
        predictions['evidence_status'] = (evidence_probs[:, 1] > evidence_threshold).int()
        
        # Evidence Extraction (Token-level argmax)
        predictions['evidence_start'] = outputs['evidence_start_logits'].argmax(dim=1)
        predictions['evidence_end'] = outputs['evidence_end_logits'].argmax(dim=1)
        
        # ESG Classification
        predictions['esg_label'] = outputs['esg_logits'].argmax(dim=1)
        
        # Timeline Classification
        predictions['timeline_label'] = outputs['timeline_logits'].argmax(dim=1)

        # Quality Classification
        predictions['quality_label'] = outputs['quality_logits'].argmax(dim=1)
        
        return predictions
    
    def span_to_text(
        self,
        text: str,
        tokens: List[int],
        offset_mapping: List[Tuple[int, int]],
        start_idx: int,
        end_idx: int
    ) -> str:
        """
        將 Token 索引範圍轉換回原文字串
        
        Args:
            text: 原始文本
            tokens: Token IDs
            offset_mapping: Tokenizer 的字元偏移映射
            start_idx: 開始 Token 索引
            end_idx: 結束 Token 索引
        
        Returns:
            對應的文本片段
        """
        # offset_mapping 可能是 Tensor，轉為 list
        if isinstance(offset_mapping, torch.Tensor):
            offset_mapping = offset_mapping.tolist()
            
        if start_idx >= len(offset_mapping) or end_idx > len(offset_mapping):
            return ""
        
        # 取得開始位置的字元位置
        start_char = offset_mapping[start_idx][0]
        # 取得結束位置的字元位置
        # 注意：我們使用 [start, end) 區間
        end_idx_clamped = min(end_idx, len(offset_mapping) - 1)
        end_char = offset_mapping[end_idx_clamped][1]
        
        if start_char == 0 and end_char == 0 and start_idx != 0:
            return ""
            
        if start_char >= len(text):
            return ""
            
        return text[start_char:min(end_char, len(text))]
    
    def inference_on_dataset(
        self,
        test_dataset: ESGDataset,
        batch_size: int = 8,
        promise_threshold: float = 0.5,
        evidence_threshold: float = 0.5
    ) -> List[Dict]:
        """
        在整個測試集上進行推理
        
        Args:
            test_dataset: 測試資料集
            batch_size: 批次大小
            promise_threshold: 承諾檢測閾值
            evidence_threshold: 證據檢測閾值
        
        Returns:
            預測結果列表
        """
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="推理中"):
                # 取得批次預測
                batch_predictions = self.predict_batch(
                    batch,
                    promise_threshold=promise_threshold,
                    evidence_threshold=evidence_threshold
                )
                
                # 轉換為 CPU 並轉換為相應格式
                batch_size_actual = batch['input_ids'].shape[0]
                
                for i in range(batch_size_actual):
                    # 提取文字片段
                    promise_str = ""
                    evidence_str = ""
                    
                    is_promise = batch_predictions['promise_status'][i].item() == 1
                    is_evidence = batch_predictions['evidence_status'][i].item() == 1
                    
                    if 'offset_mapping' in batch:
                        offsets = batch['offset_mapping'][i]
                        text_raw = batch['text'][i]
                        tokens = batch['input_ids'][i]
                        
                        if is_promise:
                            p_start = batch_predictions['promise_start'][i].item()
                            p_end = batch_predictions['promise_end'][i].item()
                            promise_str = self.span_to_text(text_raw, tokens, offsets, p_start, p_end)
                        
                        if is_evidence:
                            e_start = batch_predictions['evidence_start'][i].item()
                            e_end = batch_predictions['evidence_end'][i].item()
                            evidence_str = self.span_to_text(text_raw, tokens, offsets, e_start, e_end)

                    # ID 處理
                    sample_id = batch['id'][i]
                    if isinstance(sample_id, torch.Tensor):
                        sample_id = sample_id.item()
                    
                    # 規則：如果無承諾，timeline 必須是 N/A
                    timeline_val = self.timeline_labels.get(batch_predictions['timeline_label'][i].item(), 'N/A')
                    if not is_promise:
                        timeline_val = 'N/A'
                        promise_str = ""
                        
                    # 規則：如果無證據，quality 必須是 N/A
                    quality_val = self.quality_labels.get(batch_predictions['quality_label'][i].item(), 'N/A')
                    if not is_evidence:
                        quality_val = 'N/A'
                        evidence_str = ""

                    pred_dict = {
                        'index': sample_id,
                        'URL': batch['url'][i],
                        'page_number': batch['page_number'][i] if isinstance(batch['page_number'][i], int) else batch['page_number'][i].item(),
                        'data': batch['text'][i],
                        'ESG_type': self.esg_labels.get(batch_predictions['esg_label'][i].item(), 'S'),
                        'promise_status': 'Yes' if is_promise else 'No',
                        'promise_string': promise_str,
                        'verification_timeline': timeline_val,
                        'evidence_status': 'Yes' if is_evidence else 'No',
                        'evidence_string': evidence_str,
                        'evidence_quality': quality_val,
                    }
                    
                    all_predictions.append(pred_dict)
        
        return all_predictions
    
    def export_predictions_to_csv(
        self,
        predictions: List[Dict],
        output_path: str = "predictions.csv"
    ):
        """
        將預測結果匯出為 CSV 格式
        
        Args:
            predictions: 預測結果列表
            output_path: 輸出檔案路徑
        """
        df = pd.DataFrame(predictions)
        
        # 按照用戶要求的確切順序排列欄位
        required_columns = [
            'index', 'URL', 'page_number', 'data', 'ESG_type', 
            'promise_status', 'promise_string', 'verification_timeline', 
            'evidence_status', 'evidence_string', 'evidence_quality'
        ]
        
        # 只保留必要欄位並排序
        df = df[required_columns]
        
        # 競賽規範：UTF-8（無 BOM）、Unix 換行符 (\n)
        df.to_csv(output_path, index=False, encoding='utf-8', lineterminator='\n')
        print(f"✅ 預測結果已保存: {output_path}")
        print(f"預測數量: {len(df)}")
    
    def export_predictions_to_json(
        self,
        predictions: List[Dict],
        output_path: str = "predictions.json"
    ):
        """
        將預測結果匯出為 JSON 格式
        
        Args:
            predictions: 預測結果列表
            output_path: 輸出檔案路徑
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 預測結果已保存: {output_path}")
        print(f"預測數量: {len(predictions)}")


class ThresholdOptimizer:
    """
    最佳閾值搜尋器
    
    競賽中，調整承諾和證據檢測的概率閾值（不使用固定的 0.5）
    往往能顯著提升 F1-Score
    """
    
    def __init__(self, inference_engine: ESGInference):
        """
        初始化閾值優化器
        
        Args:
            inference_engine: 推理引擎
        """
        self.inference_engine = inference_engine
    
    def find_optimal_threshold(
        self,
        val_dataset: ESGDataset,
        batch_size: int = 8,
        thresholds: List[float] = None
    ) -> Dict[str, float]:
        """
        搜尋最佳閾值
        
        Args:
            val_dataset: 驗證資料集
            batch_size: 批次大小
            thresholds: 要搜尋的閾值列表
        
        Returns:
            最佳閾值字典
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        best_f1 = 0.0
        best_thresholds = {'promise': 0.5, 'evidence': 0.5}
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print("\n" + "=" * 80)
        print("🔍 搜尋最佳閾值")
        print("=" * 80)
        
        for promise_thresh in thresholds:
            for evidence_thresh in thresholds:
                # 計算相應的 F1-Score
                # （實際實現需要在驗證集上計算）
                # 這裡是簡化版本
                
                print(f"Promise: {promise_thresh:.1f}, Evidence: {evidence_thresh:.1f}")
        
        print("=" * 80)
        return best_thresholds
