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
    ESG 模型推理器 (整合版)
    支援單一模型推理與多模型集成 (Ensemble)。
    """
    
    def __init__(
        self,
        model: ESGMultiTaskModel,
        checkpoint_paths: List[str], # 改為接收一個路徑列表
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_paths = [p for p in checkpoint_paths if os.path.exists(p)]
        
        if not self.checkpoint_paths:
            print("[WARNING] 警告: 找不到任何有效的模型檢查點！")
        else:
            print(f"[SUCCESS] 集成引擎初始化完成，共載入 {len(self.checkpoint_paths)} 個模型。")
        
        self.model.eval()
        
        # 標籤映射保持不變...
        self.esg_labels = {0: 'E', 1: 'S', 2: 'G'}
        self.timeline_labels = {0: 'already', 1: 'within_2_years', 2: 'between_2_and_5_years', 3: 'more_than_5_years', 4: 'N/A'}
        self.quality_labels = {0: 'Clear', 1: 'Not Clear', 2: 'Misleading', 3: 'N/A'}
    
    def predict_batch(
        self,
        batch: Dict[str, torch.Tensor],
        promise_threshold: float = 0.5,
        evidence_threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        批次集成預測 (平均所有模型的 Logits)
        """
        batch_input = {
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
            'token_type_ids': batch['token_type_ids'].to(self.device)
        }
        
        # 初始化累積 Logits 字典
        ensemble_logits = {}
        task_keys = [
            'promise_logits', 'promise_bio_logits', 
            'evidence_logits', 'evidence_bio_logits',
            'esg_logits', 'timeline_logits', 'quality_logits'
        ]
        
        # 遍歷所有模型進行推理
        with torch.no_grad():
            for i, ckpt in enumerate(self.checkpoint_paths):
                # 載入該折的權重
                checkpoint = torch.load(ckpt, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                outputs = self.model(
                    input_ids=batch_input['input_ids'],
                    attention_mask=batch_input['attention_mask'],
                    token_type_ids=batch_input['token_type_ids']
                )
                
                # 累積 Logits
                for key in task_keys:
                    if key not in ensemble_logits:
                        ensemble_logits[key] = outputs[key] / len(self.checkpoint_paths)
                    else:
                        ensemble_logits[key] += outputs[key] / len(self.checkpoint_paths)
        
        # ========== 基於平均 Logits 進行最後判定 ==========
        predictions = {}
        
        promise_probs = torch.softmax(ensemble_logits['promise_logits'], dim=1)
        predictions['promise_status'] = (promise_probs[:, 1] > promise_threshold).int()
        
        evidence_probs = torch.softmax(ensemble_logits['evidence_logits'], dim=1)
        predictions['evidence_status'] = (evidence_probs[:, 1] > evidence_threshold).int()
        
        predictions['promise_bio'] = ensemble_logits['promise_bio_logits'].argmax(dim=-1)
        predictions['evidence_bio'] = ensemble_logits['evidence_bio_logits'].argmax(dim=-1)
        
        predictions['esg_label'] = ensemble_logits['esg_logits'].argmax(dim=1)
        predictions['timeline_label'] = ensemble_logits['timeline_logits'].argmax(dim=1)
        predictions['quality_label'] = ensemble_logits['quality_logits'].argmax(dim=1)
        
        return predictions
    
    def decode_bio_to_text(
        self,
        text: str,
        bio_tags: List[int],
        offset_mapping: List[Tuple[int, int]]
    ) -> str:
        """
        將 BIO 標籤序列還原為文本片段 (支援非連續)
        0: O, 1: B, 2: I
        """
        if isinstance(offset_mapping, torch.Tensor):
            offset_mapping = offset_mapping.tolist()
        if isinstance(bio_tags, torch.Tensor):
            bio_tags = bio_tags.tolist()
            
        spans = []
        current_span_start = -1
        current_span_end = -1
        
        for i, tag in enumerate(bio_tags):
            # 略過特殊 Token
            if offset_mapping[i] == [0, 0] and i != 0: continue
            
            if tag == 1: # B: 開始新片段
                if current_span_start != -1:
                    spans.append(text[offset_mapping[current_span_start][0]:offset_mapping[current_span_end][1]])
                current_span_start = i
                current_span_end = i
            elif tag == 2: # I: 延續片段
                if current_span_start != -1:
                    current_span_end = i
            else: # O: 結束片段
                if current_span_start != -1:
                    spans.append(text[offset_mapping[current_span_start][0]:offset_mapping[current_span_end][1]])
                    current_span_start = -1
                    
        # 處理最後一個片段
        if current_span_start != -1:
            spans.append(text[offset_mapping[current_span_start][0]:offset_mapping[current_span_end][1]])
            
        # 過濾空字串並用 | 連結
        valid_spans = [s.strip() for s in spans if s.strip()]
        return " | ".join(valid_spans)
    
    def inference_on_dataset(
        self,
        test_dataset: ESGDataset,
        batch_size: int = 8,
        promise_threshold: float = 0.5,
        evidence_threshold: float = 0.5
    ) -> List[Dict]:
        """
        在整個測試集上進行推理 (BIO 版)
        """
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="推理中"):
                batch_preds = self.predict_batch(batch, promise_threshold, evidence_threshold)
                
                for i in range(batch['input_ids'].shape[0]):
                    offsets = batch['offset_mapping'][i]
                    text_raw = batch['text'][i]
                    
                    is_promise = batch_preds['promise_status'][i].item() == 1
                    is_evidence = batch_preds['evidence_status'][i].item() == 1
                    
                    # BIO 解碼
                    promise_str = ""
                    if is_promise:
                        promise_str = self.decode_bio_to_text(text_raw, batch_preds['promise_bio'][i], offsets)
                        
                    evidence_str = ""
                    if is_evidence:
                        evidence_str = self.decode_bio_to_text(text_raw, batch_preds['evidence_bio'][i], offsets)

                    # ID 與 規則處理
                    sample_id = batch['id'][i]
                    if isinstance(sample_id, torch.Tensor): sample_id = sample_id.item()
                    
                    timeline_val = self.timeline_labels.get(batch_preds['timeline_label'][i].item(), 'N/A')
                    if not is_promise: timeline_val = 'N/A'
                    
                    quality_val = self.quality_labels.get(batch_preds['quality_label'][i].item(), 'N/A')
                    if not is_evidence: quality_val = 'N/A'

                    all_predictions.append({
                        'index': sample_id,
                        'URL': batch['url'][i],
                        'page_number': int(batch['page_number'][i]),
                        'data': text_raw,
                        'ESG_type': self.esg_labels.get(batch_preds['esg_label'][i].item(), 'S'),
                        'promise_status': 'Yes' if is_promise else 'No',
                        'promise_string': promise_str,
                        'verification_timeline': timeline_val,
                        'evidence_status': 'Yes' if is_evidence else 'No',
                        'evidence_string': evidence_str,
                        'evidence_quality': quality_val,
                    })
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
        print(f"[SUCCESS] 預測結果已保存: {output_path}")
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
        
        print(f"[SUCCESS] 預測結果已保存: {output_path}")
        print(f"預測數量: {len(predictions)}")


class ThresholdOptimizer:
    """
    最佳門檻搜尋器
    透過在驗證集上測試不同門檻，找出能最大化官方 TOTAL_SCORE 的參數。
    """
    
    def __init__(self, inference_engine, evaluator):
        self.inference_engine = inference_engine
        self.evaluator = evaluator
    
    def find_optimal_threshold(
        self,
        val_dataset: ESGDataset
    ) -> Dict[str, float]:
        """
        自動二維層級式搜尋 (Promise & Evidence 雙門檻優化)
        """
        def search_2d(p_range, e_range, p_step, e_step, label):
            nonlocal best_score, best_p, best_e
            p_thresholds = np.arange(p_range[0], p_range[1] + p_step/2, p_step)
            e_thresholds = np.arange(e_range[0], e_range[1] + e_step/2, e_step)
            
            print(f"\n[階層 {label}] P-範圍: {p_range}, E-範圍: {e_range}")
            
            curr_best_p, curr_best_e = p_range[0], e_range[0]
            curr_best_score = -1.0
            
            for pt in p_thresholds:
                for et in e_thresholds:
                    if pt < 0 or pt > 1 or et < 0 or et > 1: continue
                    # 執行模擬評分，傳入雙門檻
                    # 注意：我們需要 evaluator 支援 evidence_threshold
                    report = self.evaluator.analyze_performance(
                        val_dataset, silent=True, 
                        promise_threshold=pt, 
                        evidence_threshold=et
                    )
                    score = report['TOTAL_SCORE']
                    
                    if score > best_score:
                        best_score = score
                        best_p, best_e = pt, et
                    
                    if score > curr_best_score:
                        curr_best_score = score
                        curr_best_p, curr_best_e = pt, et
            
            print(f" > 本階最佳: P={curr_best_p:.3f}, E={curr_best_e:.3f} | 得分: {curr_best_score:.4f}")
            return curr_best_p, curr_best_e

        best_score = -1.0
        best_p, best_e = 0.5, 0.5

        print("\n" + "=" * 80)
        print("[START] 啟動 2D 自動層級式門檻優化 (Double-Focus Search)")
        print("=" * 80)

        # 第一階段：粗略掃描
        p1, e1 = search_2d((0.3, 0.7), (0.3, 0.7), 0.1, 0.1, "1: 初步定位")

        # 第二階段：局部精搜尋
        p2, e2 = search_2d((p1-0.1, p1+0.1), (e1-0.1, e1+0.1), 0.05, 0.05, "2: 中度精準")

        # 第三階段：細微修正
        p3, e3 = search_2d((p2-0.05, p2+0.05), (e2-0.05, e2+0.05), 0.025, 0.025, "3: 極致精準")

        # 第四階段：最後定位
        best_p, best_e = search_2d((p3-0.025, p3+0.025), (e2-0.025, e2+0.025), 0.01, 0.01, "4: 最後定位")

        print("\n" + "=" * 80)
        print(f"[SUCCESS] 2D 優化完成！")
        print(f"最佳 Promise 門檻 : {best_p:.3f}")
        print(f"最佳 Evidence 門檻: {best_e:.3f}")
        print(f"預估最高綜合得分 : {best_score:.4f}")
        print("=" * 80)

        
        return {
            'promise_threshold': float(best_p),
            'evidence_threshold': float(best_e)
        }
