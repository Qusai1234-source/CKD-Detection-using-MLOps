import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import (
    xception,
    resnet50,
    mobilenet_v2
)
from log import get_logger

logger = get_logger("preprocessing")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def get_preprocess_fn(model_name):
    if model_name == "Xception":
        logger.info("Using Xception preprocessing")
        return xception.preprocess_input

    elif model_name == "ResNet50":
        logger.info("Using ResNet50 preprocessing")
        return resnet50.preprocess_input

    elif model_name == "MobileNetV2":
        logger.info("Using MobileNetV2 preprocessing")
        return mobilenet_v2.preprocess_input

    else:
        logger.info("Using Custom CNN preprocessing")
        return lambda x: tf.cast(x, tf.float32) / 255.0


def augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.1)
    ])


def sanity_check(dataset, name="dataset"):
    images, labels = next(iter(dataset.take(1)))

    mean = tf.reduce_mean(images)
    std = tf.math.reduce_std(images)

    logger.info(f"{name} mean: {mean.numpy():.4f}")
    logger.info(f"{name} std: {std.numpy():.4f}")

    unique, counts = np.unique(labels.numpy(), return_counts=True)
    class_dist = dict(zip(unique, counts))
    logger.info(f"{name} class distribution (sample): {class_dist}")


def preprocess(dataset,model_name,augment=False):
    
    preprocess_fn = get_preprocess_fn(model_name)

    dataset = dataset.map(
        lambda x, y: (preprocess_fn(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if augment:
        aug = augmentation_layer()
        dataset = dataset.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
