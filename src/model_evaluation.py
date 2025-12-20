import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from log import get_logger
from data_ingestion import load_data
from preprocessing import preprocess

logger = get_logger("custom_cnn_evaluation")

MODEL_PATH = "artifacts/custom_cnn/best_model"
EVAL_DIR = "artifacts/evaluation"
os.makedirs(EVAL_DIR, exist_ok=True)

CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]


def evaluate_custom_cnn(val_ds):
    logger.info("Evaluating Custom CNN model")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")

    y_true, y_pred = [], []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        preds = np.argmax(preds, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES
    )

    logger.info(f"Custom CNN Accuracy: {acc:.4f}")
    logger.info("\n" + report)
    logger.info("\nConfusion Matrix:\n" + str(cm))

    report_path = os.path.join(EVAL_DIR, "custom_cnn_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: Custom CNN\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    logger.info(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    logger.info("Starting Custom CNN evaluation")

    _, raw_val_ds = load_data()

    val_ds = preprocess(
        dataset=raw_val_ds,
        model_name="custom_cnn",
        augment=False
    )

    evaluate_custom_cnn(val_ds)

    logger.info("Custom CNN evaluation completed")
