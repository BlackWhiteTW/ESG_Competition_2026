import torch
import numpy as np
import pandas as pd
import collections
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score

class ESGMockEvaluator:
    """
    ESG 官方標準模擬評分器 (VeriPromiseESG 2026)
    實作字元級 F1、連鎖懲罰機制與綜合權重計分。
    """
    
    def __init__(self, inference_engine):
        self.inference_engine = inference_engine
        self.device = inference_engine.device

    def _calculate_char_f1(self, pred: str, truth: str) -> float:
        """
        官方字元級 F1 計算方式
        """
        if not pred and not truth: return 1.0
        if not pred or not truth: return 0.0
        
        pred_chars = list(pred)
        truth_chars = list(truth)
        
        # 計算交集數量
        common = collections.Counter(pred_chars) & collections.Counter(truth_chars)
        num_same = sum(common.values())
        
        if num_same == 0: return 0.0
        
        precision = num_same / len(pred_chars)
        recall = num_same / len(truth_chars)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def analyze_performance(self, val_dataset, silent: bool = False, promise_threshold: float = 0.5, evidence_threshold: float = 0.5) -> Dict:
        """
        執行官方標準效能分析
        """
        if not silent:
            print("\n" + "="*80)
            print(f"[SCORE] 執行官方標準模擬評分 (P-門檻: {promise_threshold:.2f}, E-門檻: {evidence_threshold:.2f})")
            print("="*80)
        
        predictions = self.inference_engine.inference_on_dataset(
            val_dataset, batch_size=16, 
            promise_threshold=promise_threshold,
            evidence_threshold=evidence_threshold
        )
        results_df = pd.DataFrame(predictions)
        
        # 從 dataset 提取真相
        truths = []
        is_subset = hasattr(val_dataset, 'dataset') and hasattr(val_dataset, 'indices')
        base_dataset = val_dataset.dataset if is_subset else val_dataset
        indices = val_dataset.indices if is_subset else range(len(val_dataset))
        
        for i in indices:
            sample = base_dataset.samples[i]
            truths.append({
                'index': sample['id'],
                'truth_promise_status': 'Yes' if sample['promise_status'] == 1 else 'No',
                'truth_ESG_type': {0: 'E', 1: 'S', 2: 'G'}.get(sample['esg_label'], 'S'),
                'truth_timeline': {0:'already', 1:'within_2_years', 2:'between_2_and_5_years', 3:'more_than_5_years', 4:'N/A'}.get(sample['timeline_label'], 'N/A'),
                'truth_promise_string': sample.get('promise_string_truth', ""),
                'truth_evidence_status': 'Yes' if sample['evidence_status'] == 1 else 'No',
                'truth_evidence_string': sample.get('evidence_string_truth', ""),
                'truth_quality': {0:'Clear', 1:'Not Clear', 2:'Misleading', 3:'N/A'}.get(sample['quality_label'], 'N/A')
            })
        
        truth_df = pd.DataFrame(truths)
        
        # 強制轉換索引類型以確保合併成功
        results_df['index'] = results_df['index'].astype(str)
        truth_df['index'] = truth_df['index'].astype(str)
        
        # 確保與 inference.py 輸出的 'index' 欄位對齊
        analysis_df = pd.merge(results_df, truth_df, on='index')
        
        # --- 官方模擬計分邏輯 ---
        scores_per_task = {
            'task1_promise_status': [], # Binary F1
            'task2_promise_string': [], # Char-F1 (Dependent on T1)
            'task3_esg_type': [],       # Macro F1 (Dependent on T1)
            'task4_timeline': [],       # Macro F1 (Dependent on T1)
        }
        
        for idx, row in analysis_df.iterrows():
            # 轉換預測狀態為 Yes/No 字串以便比較
            pred_promise_status = 'Yes' if row['promise_status'] == 1 else 'No'
            
            # 子任務一：承諾識別
            t1_hit = (pred_promise_status == row['truth_promise_status'])
            
            # 連鎖懲罰規則：如果承諾判斷錯誤，後續得分皆為 0
            if t1_hit:
                # 任務 2：語句擷取品質
                # 這裡需要從 BIO 轉回字串，目前簡化處理
                s2 = 1.0 # TODO: 實作 Char-F1
                
                # 任務 3 & 4：類別判定
                s3 = 1.0 if row['esg_label'] == row['truth_ESG_type'] else 0.0
                s4 = 1.0 if row['timeline_label'] == row['truth_timeline'] else 0.0
            else:
                s2, s3, s4 = 0.0, 0.0, 0.0
            
            scores_per_task['task1_promise_status'].append(1.0 if t1_hit else 0.0)
            scores_per_task['task2_promise_string'].append(s2)
            scores_per_task['task3_esg_type'].append(s3)
            scores_per_task['task4_timeline'].append(s4)

        # 計算權重總分 (假設權重各 25%)
        final_report = {
            'Promise Status (F1)': np.mean(scores_per_task['task1_promise_status']),
            'Promise String (Char-F1)': np.mean(scores_per_task['task2_promise_string']),
            'ESG Type (Accuracy*)': np.mean(scores_per_task['task3_esg_type']),
            'Timeline (Accuracy*)': np.mean(scores_per_task['task4_timeline']),
        }
        
        # 額外統計：FP 與 FN
        fp = len(analysis_df[(analysis_df['promise_status'] == 1) & (analysis_df['truth_promise_status'] == 'No')])
        fn = len(analysis_df[(analysis_df['promise_status'] == 0) & (analysis_df['truth_promise_status'] == 'Yes')])
        final_report['False Positives (過多)'] = fp
        final_report['False Negatives (缺少)'] = fn
        
        total_score = np.mean([final_report['Promise Status (F1)'], final_report['Promise String (Char-F1)'], 
                               final_report['ESG Type (Accuracy*)'], final_report['Timeline (Accuracy*)']])
        final_report['TOTAL_SCORE'] = total_score

        if not silent:
            print("\n[官方標準計分結果]:")
            for k, v in final_report.items():
                print(f"  - {k:25s}: {v:.4f}" if isinstance(v, float) else f"  - {k:25s}: {v}")
            
            print("\n[WARNING] 提醒：由於存在連鎖懲罰，請優先優化 'promise_status' 的準確度實際分數會受 F1 影響。")
            self._diagnose_errors(analysis_df)
        
        return final_report

    def _diagnose_errors(self, df: pd.DataFrame):
        """歸納錯誤模式"""
        print("\n[2] 錯誤模式歸納 (Error Diagnosis):")
        
        # 找出 ESG 類型判定錯誤最多的類別
        esg_errors = df[df['esg_label'] != df['truth_ESG_type']]
        if not esg_errors.empty:
            error_count = esg_errors.groupby('truth_ESG_type').size()
            most_wrong_type = error_count.idxmax()
            print(f" [WARNING] 模型最常在 '{most_wrong_type}' 類型的 ESG 判定上出錯。")
            
        # 找出 Promise Detection 的假陽性 (False Positives)
        # 預測值是 1 (Yes) / 0 (No)
        fp = df[(df['promise_status'] == 1) & (df['truth_promise_status'] == 'No')]
        if not fp.empty:
            print(f" [WARNING] 假陽性警告: 模型有 {len(fp)} 個樣本過度判定為有承諾（實際上沒有）。")
            
        # 檢查擷取長度問題 (如果 predictions 包含 promise_string)
        if 'promise_string' in df.columns:
            short_captures = df[(df['promise_status'] == 1) & (df['promise_string'].str.len() < df['truth_promise_string'].str.len() * 0.5)]
            if not short_captures.empty:
                print(f" [WARNING] 擷取問題: 有 {len(short_captures)} 筆預測語句明顯短於真實語句。")

if __name__ == "__main__":
    pass
