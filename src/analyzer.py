import numpy as np
import pandas as pd
import collections
from typing import Dict


class ESGMockEvaluator:
    """
    模擬評分器
    實作字元級 F1、連鎖懲罰機制與綜合權重計分。
    """

    def __init__(self, inference_engine):
        self.inference_engine = inference_engine
        self.device = inference_engine.device

    def _calculate_char_f1(self, pred: str, truth: str) -> float:
        """
        官方字元級 F1 計算方式
        """
        if not pred and not truth:
            return 1.0
        if not pred or not truth:
            return 0.0

        pred_chars = list(pred)
        truth_chars = list(truth)

        # 計算交集數量
        common = collections.Counter(pred_chars) & collections.Counter(truth_chars)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_chars)
        recall = num_same / len(truth_chars)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def analyze_performance(
        self,
        val_dataset,
        silent: bool = False,
        promise_threshold: float = 0.5,
        evidence_threshold: float = 0.5,
    ) -> Dict:
        if not silent:
            print("\n" + "=" * 80)
            print(
                f"[SCORE] 執行官方標準模擬評分 (P-門檻: {promise_threshold:.2f}, E-門檻: {evidence_threshold:.2f})"
            )
            print("=" * 80)

        predictions = self.inference_engine.inference_on_dataset(
            val_dataset,
            batch_size=16,
            promise_threshold=promise_threshold,
            evidence_threshold=evidence_threshold,
        )
        results_df = pd.DataFrame(predictions)

        # 從 dataset 提取真相
        truths = []
        is_subset = hasattr(val_dataset, "dataset") and hasattr(val_dataset, "indices")
        base_dataset = val_dataset.dataset if is_subset else val_dataset
        indices = val_dataset.indices if is_subset else range(len(val_dataset))

        for i in indices:
            sample = base_dataset.samples[i]
            truths.append(
                {
                    "index": sample["id"],
                    "truth_promise_status": (
                        "Yes" if sample["promise_status"] == 1 else "No"
                    ),
                    "truth_ESG_type": {0: "E", 1: "S", 2: "G"}.get(
                        sample["esg_label"], "S"
                    ),
                    "truth_timeline": {
                        0: "already",
                        1: "within_2_years",
                        2: "between_2_and_5_years",
                        3: "more_than_5_years",
                        4: "N/A",
                    }.get(sample["timeline_label"], "N/A"),
                    "truth_promise_string": sample.get("promise_string_truth", ""),
                    "truth_evidence_status": (
                        "Yes" if sample["evidence_status"] == 1 else "No"
                    ),
                    "truth_evidence_string": sample.get("evidence_string_truth", ""),
                    "truth_quality": {
                        0: "Clear",
                        1: "Not Clear",
                        2: "Misleading",
                        3: "N/A",
                    }.get(sample["quality_label"], "N/A"),
                }
            )

        truth_df = pd.DataFrame(truths)

        # 強制轉換索引類型以確保合併成功
        results_df["index"] = results_df["index"].astype(str)
        truth_df["index"] = truth_df["index"].astype(str)

        # 確保與 inference.py 輸出的 'index' 欄位對齊
        analysis_df = pd.merge(results_df, truth_df, on="index")

        # --- 官方模擬計分邏輯 ---
        scores_per_task = {
            "task1_promise_status": [],  # Binary F1
            "task2_promise_string": [],  # Char-F1 (Dependent on T1)
            "task3_esg_type": [],  # Macro F1 (Dependent on T1)
            "task4_timeline": [],  # Macro F1 (Dependent on T1)
        }

        for idx, row in analysis_df.iterrows():
            # 轉換預測狀態為 Yes/No 字串以便比較
            pred_promise_status = "Yes" if row["promise_status"] == 1 else "No"

            # 子任務一：承諾識別
            t1_hit = pred_promise_status == row["truth_promise_status"]

            # 連鎖懲罰規則：如果承諾判斷錯誤，後續得分皆為 0
            if t1_hit:
                # 任務 2：語句擷取品質 (實作真實 Char-F1)
                p_str = self.inference_engine.decode_bio_to_string(
                    row["promise_bio"], row["input_ids"]
                )
                s2 = self._calculate_char_f1(p_str, row["truth_promise_string"])

                # 任務 3 & 4：類別判定
                s3 = 1.0 if row["esg_label"] == row["truth_ESG_type"] else 0.0
                s4 = 1.0 if row["timeline_label"] == row["truth_timeline"] else 0.0
            else:
                s2, s3, s4 = 0.0, 0.0, 0.0

            scores_per_task["task1_promise_status"].append(1.0 if t1_hit else 0.0)
            scores_per_task["task2_promise_string"].append(s2)
            scores_per_task["task3_esg_type"].append(s3)
            scores_per_task["task4_timeline"].append(s4)

        # 計算權重總分 (假設權重各 25%)
        final_report = {
            "Promise Status (F1)": np.mean(scores_per_task["task1_promise_status"]),
            "Promise String (Char-F1)": np.mean(
                scores_per_task["task2_promise_string"]
            ),
            "ESG Type (Accuracy*)": np.mean(scores_per_task["task3_esg_type"]),
            "Timeline (Accuracy*)": np.mean(scores_per_task["task4_timeline"]),
        }

        # 額外統計：FP 與 FN
        fp = len(
            analysis_df[
                (analysis_df["promise_status"] == 1)
                & (analysis_df["truth_promise_status"] == "No")
            ]
        )
        fn = len(
            analysis_df[
                (analysis_df["promise_status"] == 0)
                & (analysis_df["truth_promise_status"] == "Yes")
            ]
        )
        final_report["False Positives (過多)"] = fp
        final_report["False Negatives (缺少)"] = fn

        total_score = np.mean(
            [
                final_report["Promise Status (F1)"],
                final_report["Promise String (Char-F1)"],
                final_report["ESG Type (Accuracy*)"],
                final_report["Timeline (Accuracy*)"],
            ]
        )
        final_report["TOTAL_SCORE"] = total_score

        if not silent:
            print("\n[官方標準計分結果]:")
            for k, v in final_report.items():
                print(
                    f"  - {k:25s}: {v:.4f}"
                    if isinstance(v, float)
                    else f"  - {k:25s}: {v}"
                )

            print(
                "\n[WARNING] 提醒：由於存在連鎖懲罰，請優先優化 'promise_status' 的準確度實際分數會受 F1 影響。"
            )
            self._diagnose_errors(analysis_df)

        return final_report

    def _diagnose_errors(self, df: pd.DataFrame):
        """極詳細錯誤模式歸納"""
        print("\n" + "=" * 30)
        print("[2] 深度數據分析 (Detailed Diagnosis)")
        print("=" * 30)

        # 1. 承諾與證據的總體分布
        len(df)
        p_truth_count = len(df[df["truth_promise_status"] == "Yes"])
        p_pred_count = len(df[df["promise_status"] == 1])
        e_truth_count = len(df[df["truth_evidence_status"] == "Yes"])
        e_pred_count = len(df[df["evidence_status"] == 1])

        print("\n[核心判定分佈]:")
        print(
            f" - 承諾 (Promise) : 真實={p_truth_count}, 預測={p_pred_count} | 淨偏差: {p_pred_count - p_truth_count:+d}"
        )
        print(
            f" - 證據 (Evidence): 真實={e_truth_count}, 預測={e_pred_count} | 淨偏差: {e_pred_count - e_truth_count:+d}"
        )

        # 2. FP / FN 詳情
        p_fp = len(
            df[(df["promise_status"] == 1) & (df["truth_promise_status"] == "No")]
        )
        p_fn = len(
            df[(df["promise_status"] == 0) & (df["truth_promise_status"] == "Yes")]
        )
        e_fp = len(
            df[(df["evidence_status"] == 1) & (df["truth_evidence_status"] == "No")]
        )
        e_fn = len(
            df[(df["evidence_status"] == 0) & (df["truth_evidence_status"] == "Yes")]
        )

        print("\n[錯誤細節 (FP/FN)]:")
        print(f" - 承諾判定: 多抓(FP)={p_fp:3d} 筆, 漏抓(FN)={p_fn:3d} 筆")
        print(f" - 證據判定: 多抓(FP)={e_fp:3d} 筆, 漏抓(FN)={e_fn:3d} 筆")

        # 3. ESG 類型詳細統計 (僅針對有承諾的樣本)
        print("\n[ESG 類型分佈 (預測 vs 真實)]:")
        for t in ["E", "S", "G"]:
            t_truth = len(df[df["truth_ESG_type"] == t])
            t_pred = len(df[df["esg_label"] == t])
            t_correct = len(df[(df["esg_label"] == t) & (df["truth_ESG_type"] == t)])
            acc = (t_correct / t_truth * 100) if t_truth > 0 else 0
            print(
                f" - 類型 {t}: 預測={t_pred:3d}, 真實={t_truth:3d} | 答對={t_correct:3d} ({acc:.1f}%)"
            )

        # 4. 時程判定統計
        print("\n[時程判定 (Timeline) 準確率]:")
        timeline_correct = len(
            df[
                (df["timeline_label"] == df["truth_timeline"])
                & (df["truth_promise_status"] == "Yes")
            ]
        )
        print(f" - 正確判定: {timeline_correct} / {p_truth_count} (僅計真實有承諾者)")

        print("\n" + "=" * 30)


if __name__ == "__main__":
    pass
