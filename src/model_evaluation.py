import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from log import get_logger
from data_ingestion import load_data
from preprocessing import preprocess

logger = get_logger("model_evaluation")

ARTIFACTS_DIR = "artifacts"
EVAL_DIR = os.path.join(ARTIFACTS_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]
MODELS = ["Xception", "ResNet50", "MobileNetV2", "CustomCNN"]


def evaluate_model(model_name, val_ds):
    logger.info(f"Evaluating model: {model_name}")

    model_path = os.path.join(ARTIFACTS_DIR, model_name, "best_model")
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return

    model = tf.keras.models.load_model(model_path)
    logger.info(f"Loaded model from {model_path}")

    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        preds = np.argmax(preds, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES
    )

    logger.info(f"{model_name} Accuracy: {acc:.4f}")

    # Save report
    report_path = os.path.join(EVAL_DIR, f"{model_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    logger.info(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    logger.info("Starting model evaluation pipeline")

    _, val_ds = load_data()

    val_ds = preprocess(
        val_ds,
        augment=False,
        balance=False
    )

    for model_name in MODELS:
        evaluate_model(model_name, val_ds)

    logger.info("Model evaluation completed successfully")
