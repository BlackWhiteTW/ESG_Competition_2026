import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Dict, List, Union
from tqdm import tqdm

import config
from model import ESGMultiTaskModel


class ESGInference:
    """
    ESG 模型推理器 (整合版)
    支援單一模型推理與多模型集成 (Ensemble)。
    """

    def __init__(
        self,
        model: ESGMultiTaskModel,
        checkpoint_paths: Union[str, List[str]],
        model_name: str = config.MODEL_NAME,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 標準化為列表格式
        if isinstance(checkpoint_paths, str):
            self.checkpoint_paths = [checkpoint_paths]
        else:
            self.checkpoint_paths = [p for p in checkpoint_paths if os.path.exists(p)]

        # 預先載入所有權重到系統記憶體 (CPU) 中，避免 VRAM 爆炸
        self.model_weights = []
        for ckpt in self.checkpoint_paths:
            if os.path.exists(ckpt):
                # 關鍵修改：map_location="cpu"
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
                self.model_weights.append(state["model_state_dict"])

        if not self.model_weights:
            print("[WARNING] 警告: 找不到任何有效的模型檢查點！")
        else:
            print(
                f"[SUCCESS] 集成引擎初始化完成，共載入 {len(self.model_weights)} 個模型權重。"
            )

        self.model.eval()

        self.esg_labels = {0: "E", 1: "S", 2: "G"}
        self.timeline_labels = {
            0: "already",
            1: "within_2_years",
            2: "between_2_and_5_years",
            3: "more_than_5_years",
            4: "N/A",
        }
        self.quality_labels = {0: "Clear", 1: "Not Clear", 2: "Misleading", 3: "N/A"}

    def predict_batch(
        self,
        batch: Dict[str, torch.Tensor],
        promise_threshold: float = 0.5,
        evidence_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        批次集成預測
        """
        batch_input = {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "token_type_ids": batch["token_type_ids"].to(self.device),
        }

        ensemble_logits = {}
        task_keys = [
            "promise_logits",
            "promise_bio_logits",
            "evidence_logits",
            "evidence_bio_logits",
            "esg_logits",
            "timeline_logits",
            "quality_logits",
        ]

        num_models = len(self.model_weights)

        with torch.no_grad():
            if num_models > 0:
                # 集成模式：平均多個權重的輸出
                for state_dict in self.model_weights:
                    self.model.load_state_dict(state_dict)
                    outputs = self.model(
                        input_ids=batch_input["input_ids"],
                        attention_mask=batch_input["attention_mask"],
                        token_type_ids=batch_input["token_type_ids"],
                    )

                    for key in task_keys:
                        if key not in ensemble_logits:
                            ensemble_logits[key] = outputs[key] / num_models
                        else:
                            ensemble_logits[key] += outputs[key] / num_models
            else:
                # 單模型模式：直接使用當前模型的權重 (用於訓練中評估)
                outputs = self.model(
                    input_ids=batch_input["input_ids"],
                    attention_mask=batch_input["attention_mask"],
                    token_type_ids=batch_input["token_type_ids"],
                )
                for key in task_keys:
                    ensemble_logits[key] = outputs[key]

        # 判定邏輯
        predictions = {}
        promise_probs = torch.softmax(ensemble_logits["promise_logits"], dim=1)
        predictions["promise_status"] = (promise_probs[:, 1] > promise_threshold).int()

        evidence_probs = torch.softmax(ensemble_logits["evidence_logits"], dim=1)
        predictions["evidence_status"] = (
            evidence_probs[:, 1] > evidence_threshold
        ).int()

        predictions["promise_bio"] = ensemble_logits["promise_bio_logits"].argmax(
            dim=-1
        )
        predictions["evidence_bio"] = ensemble_logits["evidence_bio_logits"].argmax(
            dim=-1
        )

        predictions["esg_label"] = ensemble_logits["esg_logits"].argmax(dim=1)
        predictions["timeline_label"] = ensemble_logits["timeline_logits"].argmax(dim=1)
        predictions["quality_label"] = ensemble_logits["quality_logits"].argmax(dim=1)

        return predictions

    def inference_on_dataset(
        self, dataset, promise_threshold=0.5, evidence_threshold=0.5, batch_size=8
    ):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_results = []
        for batch in tqdm(dataloader, desc="推理中"):
            preds = self.predict_batch(batch, promise_threshold, evidence_threshold)

            # 將 Tensor 轉為 List 以便處理
            curr_batch_size = len(batch["input_ids"])
            for i in range(curr_batch_size):
                # 處理 index 可能為 tensor 的情況
                idx = batch["id"][i]
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()

                result = {
                    "index": idx,
                    "url": batch.get("url", [""] * curr_batch_size)[i],
                    "page_number": batch.get("page_number", [0] * curr_batch_size)[
                        i
                    ].item()
                    if isinstance(
                        batch.get("page_number", [0] * curr_batch_size)[i], torch.Tensor
                    )
                    else batch.get("page_number", [0] * curr_batch_size)[i],
                    "data": batch.get("text", [""] * curr_batch_size)[i],
                    "input_ids": batch["input_ids"][i].cpu().numpy(),
                    "promise_status": preds["promise_status"][i].item(),
                    "evidence_status": preds["evidence_status"][i].item(),
                    "esg_label": self.esg_labels[preds["esg_label"][i].item()],
                    "timeline_label": self.timeline_labels[
                        preds["timeline_label"][i].item()
                    ],
                    "quality_label": self.quality_labels[
                        preds["quality_label"][i].item()
                    ],
                    "promise_bio": preds["promise_bio"][i].cpu().numpy(),
                    "evidence_bio": preds["evidence_bio"][i].cpu().numpy(),
                }
                all_results.append(result)
        return all_results

    def decode_bio_to_string(self, bio_tags, input_ids):
        """
        將 BIO 序列標記轉換回原始字串 (針對中文優化)。
        """
        selected_tokens = []
        for tag, token_id in zip(bio_tags, input_ids):
            if tag in [1, 2]:  # B (1) or I (2)
                if token_id not in [
                    self.tokenizer.cls_token_id,
                    self.tokenizer.sep_token_id,
                    self.tokenizer.pad_token_id,
                ]:
                    selected_tokens.append(int(token_id))

        if not selected_tokens:
            return ""

        # 解碼並處理中文常見的 ## 符號與空格
        decoded_text = self.tokenizer.decode(selected_tokens)
        decoded_text = decoded_text.replace(" ", "").replace("##", "")
        return decoded_text.strip()

    def export_predictions_to_csv(self, predictions: List[Dict], output_file: str):
        """
        將推理結果匯出為 CSV 格式 (嚴格匹配官方 [External] 範本)。
        """
        rows = []
        for p in predictions:
            # 解碼
            p_str = (
                self.decode_bio_to_string(p["promise_bio"], p["input_ids"])
                if p["promise_status"] == 1
                else ""
            )
            e_str = (
                self.decode_bio_to_string(p["evidence_bio"], p["input_ids"])
                if p["evidence_status"] == 1
                else ""
            )

            # 依照官方範本精準映射:
            # index, URL, paage_number, data, ESG_type, promise_status, promise_string, verification_timeline, evidence_status, evidence_string, evidence_quality
            rows.append(
                {
                    "index": p["index"],
                    "URL": p.get("url", ""),
                    "paage_number": p.get("page_number", 0),
                    "data": p.get("data", ""),
                    "ESG_type": p["esg_label"],
                    "promise_status": "Yes" if p["promise_status"] == 1 else "No",
                    "promise_string": p_str,
                    "verification_timeline": p["timeline_label"]
                    if p["promise_status"] == 1
                    else "N/A",
                    "evidence_status": "Yes" if p["evidence_status"] == 1 else "No",
                    "evidence_string": e_str if p["evidence_status"] == 1 else "",
                    "evidence_quality": p["quality_label"]
                    if p["evidence_status"] == 1
                    else "N/A",
                }
            )

        export_df = pd.DataFrame(rows)
        # 強制指定欄位順序 (包含那個拼錯的 paage_number)
        official_columns = [
            "index",
            "URL",
            "paage_number",
            "data",
            "ESG_type",
            "promise_status",
            "promise_string",
            "verification_timeline",
            "evidence_status",
            "evidence_string",
            "evidence_quality",
        ]
        export_df = export_df[official_columns]

        # 存檔 (使用 utf-8-sig 確保 Excel 開啟不掉字)
        export_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"[SUCCESS] 格式修正完畢！已儲存至: {output_file}")

    def export_predictions_to_json(self, predictions: List[Dict], output_file: str):
        """
        將推理結果匯出為 JSON 格式。
        """
        with open(output_file, "w", encoding="utf-8") as f:
            # 轉換 numpy 陣列以便 JSON 序列化
            serializable_preds = []
            for p in predictions:
                new_p = p.copy()
                if isinstance(new_p.get("promise_bio"), np.ndarray):
                    new_p["promise_bio"] = new_p["promise_bio"].tolist()
                if isinstance(new_p.get("evidence_bio"), np.ndarray):
                    new_p["evidence_bio"] = new_p["evidence_bio"].tolist()
                serializable_preds.append(new_p)
            json.dump(serializable_preds, f, ensure_ascii=False, indent=4)
        print(f"[SUCCESS] JSON 結果已儲存至: {output_file}")


class ThresholdOptimizer:
    def __init__(self, inference_engine, evaluator):
        self.engine = inference_engine
        self.evaluator = evaluator

    def find_optimal_threshold(self, dataset):
        print("\n" + "=" * 80)
        print("[START] 啟動 2D 全域掃描門檻優化 (Grid Search)")
        print("=" * 80)

        best_score = -1.0
        best_p, best_e = 0.5, 0.5

        # 擴大搜尋範圍與精度 (從 0.1 到 0.9)
        p_values = np.linspace(0.1, 0.9, 17)  # 步長約 0.05
        e_values = np.linspace(0.1, 0.9, 17)

        print(f"正在搜尋 {len(p_values) * len(e_values)} 組門檻組合...")

        for p in tqdm(p_values, desc="優化 Promise"):
            for e in e_values:
                # 執行靜默分析
                report = self.evaluator.analyze_performance(
                    dataset, silent=True, promise_threshold=p, evidence_threshold=e
                )
                score = report["TOTAL_SCORE"]

                # 加入 FP 懲罰邏輯：如果 FP 過高，稍微扣分以防出現全猜 Yes 的極端情況
                fp_penalty = (report["False Positives (過多)"] / len(dataset)) * 0.1
                adjusted_score = score - fp_penalty

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_p, best_e = p, e

        # 最終確認最佳結果
        final_report = self.evaluator.analyze_performance(
            dataset, silent=False, promise_threshold=best_p, evidence_threshold=best_e
        )

        print("\n" + "=" * 80)
        print(
            f"[SUCCESS] 優化完成！最佳組合：Promise={best_p:.2f}, Evidence={best_e:.2f}"
        )
        print(f"最終預估真實總分: {final_report['TOTAL_SCORE']:.4f}")
        print("=" * 80)

        return {
            "promise_threshold": float(best_p),
            "evidence_threshold": float(best_e),
            "estimated_score": float(final_report["TOTAL_SCORE"]),
        }
