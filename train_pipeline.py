import os
import time
import logging
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
from ultralytics import YOLO

# ==================== 1. Configuration ====================
DATASET_ROOT = r"/public/home/yangh/.git/CigYOLO/dataset"
IMG_SIZE = 800

LABEL_WRONG = 1
LABEL_RIGHT = 0

DEVICE = 0 if torch.cuda.is_available() else 'cpu'
print(f"Current device: {'GPU (cuda:0)' if DEVICE == 0 else 'CPU'}")

N_TRIALS = 50
EPOCHS_PER_TRIAL = 15
FINAL_EPOCHS = 100

OUTPUT_LOG_DIR = os.path.join(DATASET_ROOT, "pipeline_outputs")
HPO_WEIGHTS_DIR = os.path.join(OUTPUT_LOG_DIR, "hpo_weights")
FINAL_MODELS_DIR = os.path.join(OUTPUT_LOG_DIR, "final_models")
os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
os.makedirs(HPO_WEIGHTS_DIR, exist_ok=True)
os.makedirs(FINAL_MODELS_DIR, exist_ok=True)

# ==================== 2. Image Channel Verification ====================
def verify_and_fix_channels(root_dir):
    print("Performing image channel security verification on 800x600 images...")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            continue

        for category in ['Wrong', 'Right']:
            cat_path = os.path.join(split_path, category)
            if not os.path.exists(cat_path):
                continue

            files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg'))]
            for file in tqdm(files, desc=f"Verifying {split}/{category}"):
                img_path = os.path.join(cat_path, file)
                try:
                    with Image.open(img_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                            img.save(img_path, quality=98)
                except Exception as e:
                    print(f"Corrupted file skipped: {img_path}, error: {e}")

# ==================== 3. Evaluation Function ====================
def evaluate_dataset(model, root_dir, split_name):
    print(f"\nEvaluating [{split_name}] set with full metrics (Wrong=1, Right=0)...")
    target_dir = os.path.join(root_dir, split_name)

    y_true = []
    y_pred = []

    for category, target_label in [('Wrong', LABEL_WRONG), ('Right', LABEL_RIGHT)]:
        cat_path = os.path.join(target_dir, category)
        if not os.path.exists(cat_path):
            continue

        files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        if not files:
            continue

        print(f"Predicting {split_name}/{category} images...")
        for file in tqdm(files, desc=f"  [{split_name}]"):
            img_path = os.path.join(cat_path, file)
            results = model.predict(img_path, imgsz=IMG_SIZE, device=DEVICE, verbose=False)

            raw_pred = results[0].probs.top1
            y_true.append(target_label)
            y_pred.append(raw_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=LABEL_WRONG, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=LABEL_WRONG, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[LABEL_WRONG, LABEL_RIGHT])

    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]

    return {
        "Overall Accuracy": f"{acc*100:.2f}%",
        "Precision (Label 1)": f"{precision*100:.2f}%",
        "Recall (Label 1)": f"{recall*100:.2f}%",
        "Matthews Correlation Coefficient (MCC)": f"{mcc:.4f}",
        "True Positives (Actual=1, Pred=1)": tp,
        "False Negatives (Actual=1, Pred=0) [Missed Detections]": fn,
        "False Positives (Actual=0, Pred=1) [False Alarms]": fp,
        "True Negatives (Actual=0, Pred=0)": tn
    }

# ==================== 4. Loss Curve Plotting ====================
def plot_loss_curve(yolo_save_dir, output_root):
    csv_path = os.path.join(yolo_save_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"Training log file not found: {csv_path}, cannot plot loss curve.")
        return

    print("\nExtracting loss data and generating visualization...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    epochs = df['epoch']
    train_loss = df['train/loss']
    val_loss = df['val/loss']

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', linewidth=2, linestyle='-')
    plt.plot(epochs, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2, linestyle='--')

    plt.title('YOLOv11 Tobacco Classification Model - Loss Curve', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='upper right')

    plt.tight_layout()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    img_filename = f"yolo11_loss_curve_{timestamp}.png"
    save_path = os.path.join(output_root, img_filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Loss curve saved to: {save_path}")

# ==================== 5. Hyperparameter Optimization ====================
hpo_custom_records = []
best_hpo_score = -1.0

def objective(trial):
    global best_hpo_score
    print(f"\n==================== [HPO Phase] Trial {trial.number + 1}/{N_TRIALS} ====================")

    suggested_params = {
        "lr0": trial.suggest_float("lr0", 1e-4, 1e-2, log=True),
        "batch": trial.suggest_categorical("batch", [16, 32, 64]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
        "hsv_v": trial.suggest_float("hsv_v", 0.2, 0.6),
        "bgr": trial.suggest_float("bgr", 0.0, 0.2),
        "mixup": trial.suggest_float("mixup", 0.0, 0.5),
        "fliplr": trial.suggest_float("fliplr", 0.0, 1.0),
        "flipud": trial.suggest_float("flipud", 0.0, 1.0)
    }

    print(f"Suggested parameters: {suggested_params}")

    model = YOLO('yolo11n-cls.pt')
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    model.train(
        data=DATASET_ROOT,
        epochs=EPOCHS_PER_TRIAL,
        imgsz=IMG_SIZE,
        workers=4,
        device=DEVICE,
        amp=False,
        val=False,
        plots=False,
        **suggested_params
    )

    metrics = evaluate_dataset(model, DATASET_ROOT, 'val')

    acc = float(metrics["Overall Accuracy"].strip('%')) / 100
    precision = float(metrics["Precision (Label 1)"].strip('%')) / 100
    recall = float(metrics["Recall (Label 1)"].strip('%')) / 100
    mcc = float(metrics["Matthews Correlation Coefficient (MCC)"])

    p = precision
    r = recall

    beta = 2
    f2_score = 0.0 if (p + r) == 0 else (1 + beta**2) * (p * r) / ((beta**2 * p) + r)

    record = {
        "Trial Number": trial.number + 1,
        "F2-Score": f2_score,
        "Validation_Recall": r,
        "Validation_MCC": mcc,
        "Validation_Accuracy": acc,
        "Validation_Precision": p,
        "Validation_FN": int(metrics["False Negatives (Actual=1, Pred=0) [Missed Detections]"]),
        "Validation_FP": int(metrics["False Positives (Actual=0, Pred=1) [False Alarms]"])
    }
    for k, v in suggested_params.items():
        record[f"param_{k}"] = v
    hpo_custom_records.append(record)

    if f2_score > best_hpo_score:
        best_hpo_score = f2_score
        hpo_model_save_path = os.path.join(HPO_WEIGHTS_DIR, f"hpo_best_trial_{trial.number + 1}.pt")
        model.save(hpo_model_save_path)
        print(f"New best score! Model saved to: {hpo_model_save_path}")

    print(f"Trial ended -> Recall: {r*100:.2f}% | MCC: {mcc:.4f} | F2-Score: {f2_score:.4f}")
    return f2_score

def load_best_hps_from_history(df_history):
    f2_col_name = None
    for col in df_history.columns:
        if 'f2' in col.lower():
            f2_col_name = col
            break

    if not f2_col_name:
        raise KeyError("Cannot find F2-Score column")

    best_idx = df_history[f2_col_name].idxmax()
    best_row = df_history.iloc[best_idx]

    best_hps = {}
    for col in df_history.columns:
        if col.startswith("param_"):
            param_name = col.replace("param_", "")
            val = best_row[col]
            if param_name == "batch":
                best_hps[param_name] = int(val)
            else:
                best_hps[param_name] = float(val)

    return best_hps, best_row[f2_col_name]

# ==================== 6. Final Training Function ====================
def train_yolo_model(root_dir, best_hps):
    print(f"\nStarting YOLOv11 classification model final training (Input resolution: {IMG_SIZE})...")
    print("\nHyperparameter configuration:")
    for k, v in best_hps.items():
        print(f"   {k}: {v}")

    model = YOLO('yolo11n-cls.pt')

    results = model.train(
        data=root_dir,
        epochs=FINAL_EPOCHS,
        imgsz=IMG_SIZE,
        batch=best_hps.get('batch', 16),
        workers=4,
        device=DEVICE,
        patience=10,
        save_period=10,
        val=True,
        plots=True,
        hsv_v=best_hps.get('hsv_v', 0.4),
        bgr=best_hps.get('bgr', 0.2),
        fliplr=best_hps.get('fliplr', 0.5),
        flipud=best_hps.get('flipud', 0.0),
        mixup=best_hps.get('mixup', 0.0),
        label_smoothing=best_hps.get('label_smoothing', 0.0),
        lr0=best_hps.get('lr0', 0.01),
        amp=False,
    )
    save_dir_abs = os.path.abspath(results.save_dir)
    print(f"Training results saved to: {save_dir_abs}")
    return model, save_dir_abs

# ==================== 7. Existing Model Check ====================
def find_existing_final_model():
    runs_dir = os.path.join(DATASET_ROOT, "runs", "classify")
    if os.path.exists(runs_dir):
        train_dirs = [d for d in os.listdir(runs_dir) if d.startswith('train')]
        if train_dirs:
            latest_dir = max(train_dirs, key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)))
            best_weights_path = os.path.join(runs_dir, latest_dir, "weights", "best.pt")
            if os.path.exists(best_weights_path):
                return best_weights_path, os.path.join(runs_dir, latest_dir)

    final_models = [f for f in os.listdir(FINAL_MODELS_DIR) if f.endswith('.pt')]
    if final_models:
        latest_model = max(final_models, key=lambda f: os.path.getmtime(os.path.join(FINAL_MODELS_DIR, f)))
        return os.path.join(FINAL_MODELS_DIR, latest_model), None

    return None, None

# ==================== 8. Main Pipeline ====================
if __name__ == "__main__":
    verify_and_fix_channels(DATASET_ROOT)

    hpo_csv_path = os.path.join(OUTPUT_LOG_DIR, "hpo_trials_performance_history.csv")
    best_hps = None
    skip_hpo = False

    print("\nChecking historical tuning records...")
    if os.path.exists(hpo_csv_path):
        try:
            encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030']
            df_history = None
            for enc in encodings:
                try:
                    df_history = pd.read_csv(hpo_csv_path, encoding=enc)
                    print(f"Successfully read CSV with {enc} encoding")
                    break
                except:
                    continue

            if df_history is not None and len(df_history) == N_TRIALS:
                print(f"Loaded optimization history ({len(df_history)} trials).")
                best_hps, best_f2 = load_best_hps_from_history(df_history)
                skip_hpo = True
                print(f"Best F2-Score from history: {best_f2:.4f}")
            else:
                print(f"History trials ({len(df_history) if df_history is not None else 0}) != target ({N_TRIALS})")
        except Exception as e:
            print(f"Failed to parse history: {e}")

    if not skip_hpo:
        print("\nStarting Optuna hyperparameter optimization...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_TRIALS)

        df_optuna = study.trials_dataframe()
        df_custom_hpo = pd.DataFrame(hpo_custom_records)
        df_hpo_history = pd.merge(
            df_optuna[['number', 'datetime_start', 'datetime_complete', 'duration', 'state']],
            df_custom_hpo, left_on='number', right_on='Trial Number'
        ).drop(columns=['number'])
        df_hpo_history.to_csv(hpo_csv_path, index=False, encoding='utf-8-sig')
        best_hps = study.best_params
        print(f"Best F2-Score from optimization: {study.best_value:.4f}")
    else:
        print("Using historical hyperparameter configuration.")

    print("\nFinal Training Hyperparameters:")
    for k, v in best_hps.items():
        print(f"   {k}: {v}")

    print("\nChecking for existing final model checkpoints...")
    existing_model_path, existing_train_dir = find_existing_final_model()
    skip_final_training = False

    if existing_model_path is not None:
        file_size = os.path.getsize(existing_model_path) / (1024 * 1024)
        print(f"Found existing model: {existing_model_path} ({file_size:.2f} MB)")

        training_complete = False
        if existing_train_dir is not None:
            results_csv = os.path.join(existing_train_dir, "results.csv")
            if os.path.exists(results_csv):
                try:
                    df_results = pd.read_csv(results_csv)
                    if len(df_results) >= FINAL_EPOCHS:
                        training_complete = True
                        print(f"Training completed: {len(df_results)}/{FINAL_EPOCHS} epochs")
                    else:
                        print(f"Training incomplete: {len(df_results)}/{FINAL_EPOCHS} epochs")
                except:
                    pass

        response = input("Skip training and use existing model? (y/n): ")
        if response.lower() == 'y':
            skip_final_training = True
            model = YOLO(existing_model_path)
            print("Using existing model for evaluation")

    if not skip_final_training:
        print("\nStarting final training phase...")
        model, yolo_save_dir = train_yolo_model(DATASET_ROOT, best_hps)

        plot_loss_curve(yolo_save_dir, OUTPUT_LOG_DIR)
        print('yolo_save_dir:   ', yolo_save_dir)

        best_weights = os.path.join(yolo_save_dir, "weights", "best.pt")
        if os.path.exists(best_weights):
            model = YOLO(best_weights)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            final_model_path = os.path.join(FINAL_MODELS_DIR, f"final_model_{timestamp}.pt")
            model.save(final_model_path)
            print(f"Final model saved to: {final_model_path}")
        else:
            print(f"\nWeight file not found at: {best_weights}")
            exit(1)
    else:
        print("Skipping training phase, using existing checkpoint.")

    print("\nGenerating final compliance report...")

    val_metrics = evaluate_dataset(model, DATASET_ROOT, 'val')
    test_metrics = evaluate_dataset(model, DATASET_ROOT, 'test')

    df_metrics = pd.DataFrame({
        "Evaluation Metric (Standard: Wrong=1, Right=0)": list(val_metrics.keys()),
        "Validation Set": list(val_metrics.values()),
        "Test Set": list(test_metrics.values())
    })

    print("\n" + "="*60)
    print("   YOLOv11 Defect Detection Report (Wrong = 1 | Right = 0)")
    print("="*60)
    print(df_metrics.to_string(index=False))
    print("="*60)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"yolo11_metrics_1_0_{timestamp}.csv"
    csv_path = os.path.join(OUTPUT_LOG_DIR, csv_filename)

    df_metrics.to_csv(csv_path, index=False, encoding='utf_8_sig')
    print(f"\nStandardized evaluation report (1/0 encoding) saved to: {csv_path}")

    print("\n" + "="*80)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*80)
    print(f"Final Validation Recall: {val_metrics['Recall (Label 1)']}")
    print(f"Final Test Recall: {test_metrics['Recall (Label 1)']}")
    print(f"Final Validation MCC: {val_metrics['Matthews Correlation Coefficient (MCC)']}")
    print(f"Final Test MCC: {test_metrics['Matthews Correlation Coefficient (MCC)']}")
    print(f"Final Validation Accuracy: {val_metrics['Overall Accuracy']}")
    print(f"Final Test Accuracy: {test_metrics['Overall Accuracy']}")
    print(f"Evaluation report saved to: {csv_path}")
    print("="*80)