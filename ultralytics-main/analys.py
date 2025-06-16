from ultralytics import YOLO
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import os
from tqdm import tqdm

def evaluate_yolo_binary(model_path, val_dir, conf_thres=0.25, iou_thres=0.45):
    model = YOLO(model_path)
    results = model.val(conf=conf_thres, iou=iou_thres)

    y_true = []
    y_pred = []
    y_prob = []

    for result in tqdm(results):
        # True labels
        gt_boxes = result.boxes  # Ground truth is inferred from path structure (optional, or via labels dir)
        
        # Inference predictions
        pred_classes = result.boxes.cls.cpu().numpy() if result.boxes else []
        pred_confs = result.boxes.conf.cpu().numpy() if result.boxes else []

        # Decide: is this image classified as positive?
        if len(pred_classes) > 0:
            y_pred.append(1)  # Predicts at least one "陨石"
            y_prob.append(float(np.max(pred_confs)))
        else:
            y_pred.append(0)
            y_prob.append(0.0)

        # You must provide true label from filename or folder structure
        # For example: val_dir/class0/xxx.jpg --> class = 0
        img_path = result.path
        true_label = 1 if "meteorite" in img_path.lower() else 0
        y_true.append(true_label)

    # Metrics
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Non-Meteorite", "Meteorite"]))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Meteorite", "Meteorite"], yticklabels=["Non-Meteorite", "Meteorite"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Confidence Histogram
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)
    plt.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="Non-Meteorite")
    plt.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label="Meteorite")
    plt.xlabel("Predicted Confidence")
    plt.ylabel("Sample Count")
    plt.legend()
    plt.title("Confidence Histogram")
    plt.grid(True)
    plt.show()

    # ROC AUC
    auc_score = roc_auc_score(y_true, y_prob)
    print(f"ROC AUC: {auc_score:.4f}")


evaluate_yolo_binary(
    model_path="/root/jzh/demo_2/ultralytics-main/runs/classify/train3/weights/best.pt", 
    val_dir="/root/jzh/meteorite/meteorite_process/val", 
    conf_thres=0.3
)
