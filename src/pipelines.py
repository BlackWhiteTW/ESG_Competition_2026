import os
import torch
from datetime import datetime
from dataset import ESGDataset
from model import ESGMultiTaskModel
from train import ESGTrainer, create_data_splits
from inference import ESGInference
from utils import print_header, print_system_info
import config


def run_train_pipeline(
    resume_checkpoint=None,
    json_file=config.DATA_PATH,
    num_epochs=10,
    batch_size=8,
    learning_rate=2e-5,
    gradient_accumulation_steps=2,
):
    """執行完整訓練流水線"""
    print_header("[START] 開始訓練流程")
    # 偵測並印出當前執行環境的系統資訊。
    print_system_info()

    # 載入資料
    dataset = ESGDataset(json_file=json_file, model_name=config.MODEL_NAME, debug=True)
    train_dataloader, val_dataloader = create_data_splits(
        dataset, train_ratio=0.8, batch_size=batch_size
    )

    # 模型與訓練器初始化
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESGMultiTaskModel(model_name=config.MODEL_NAME, dropout_rate=0.1)

    trainer = ESGTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        device=device,
        checkpoint_dir=config.MODELS_DIR,
    )

    # 開始執行
    best_checkpoint = os.path.join(trainer.checkpoint_dir, "best_model.pt")
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train(
            resume_from_checkpoint=best_checkpoint
            if os.path.exists(best_checkpoint)
            else None
        )

    return best_checkpoint


def run_inference_pipeline(
    checkpoint_path, json_file=config.DATA_PATH, output_format="csv"
):
    """執行完整推理流水線"""
    print_header("[START] 開始推理流程")
    # 偵測並印出當前執行環境的系統資訊。
    print_system_info()

    test_dataset = ESGDataset(
        json_file=json_file, model_name=config.MODEL_NAME, debug=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESGMultiTaskModel(model_name=config.MODEL_NAME)
    inference_engine = ESGInference(
        model=model, checkpoint_paths=[checkpoint_path], device=device
    )

    predictions = inference_engine.inference_on_dataset(test_dataset, batch_size=16)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"predictions_{timestamp}.{output_format}"

    # 確保輸出目錄存在
    if not os.path.exists(config.OUTPUTS_DIR):
        os.makedirs(config.OUTPUTS_DIR)

    output_path = os.path.join(config.OUTPUTS_DIR, output_file)

    if output_format == "csv":
        inference_engine.export_predictions_to_csv(predictions, output_path)
    else:
        inference_engine.export_predictions_to_json(predictions, output_path)

    print(f"[SUCCESS] 結果已匯出: {output_path}")
    return output_path
