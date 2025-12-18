import cv2
import os
from log import get_logger
from tensorflow.keras.layers import Rescaling 
import tensorflow as tf
from collections import Counter
logger=get_logger("Preprocessing")
def normalize_data(train_ds,val_ds):
    logger.debug("Normalizing train_ds and val_ds")
    normalization_layer=Rescaling(1./255)

    train_ds=train_ds.map(lambda x,y:(normalization_layer(x),y))
    val_ds=val_ds.map(lambda x,y:(normalization_layer(x),y))

    logger("Prefetching dataset")
    AUTOTUNE=tf.data.AUTOTUNE
    train_ds=train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds=val_ds.prefetch(buffer_size=AUTOTUNE)

    logger("Normalization and prefetching applied.")
    return train_ds,val_ds

def augment_data(dataset):
    aug_layer=tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1)
    ])
    dataset=dataset.map(lambda x,y:(aug_layer(x,training=True),y),num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def count_classes(dataset):

    counter=Counter()
    for _,label in dataset:
        counter[int(label.numpy())]+=1
    logger.debug(f"Class Distribution before balancing:{dict(counter)}")
    return counter

def upsample(dataset):

    logger.debug("Starting Upsampling.")
    class_counts=count_classes(dataset)
    max_count=max(class_counts.values())

    class_datasets=[]
    for class_id,count in class_counts.items():
        repeat_factor=max_count//count

        class_ds=dataset.filter(lambda x,y:y==class_id)
        class_ds=class_ds.repeat(repeat_factor)
        class_datasets.append(class_ds)
    balanced_ds=class_datasets[0]
    for ds in class_datasets[1:]:
        balanced_ds=balanced_ds.concatenate(ds)
    
    logger.debug("Sampling completed.")
    return balanced_ds

def preprocess_image(image,label):
    image=tf.cast(image,tf.float32)
    image=image/255.0
    return image,label

def preprocess(dataset,augment=False,balance=False):

    logger.debug("Starting Preprocessing.")

    dataset=dataset.unbatch()
    if balance:
        balanced_ds=upsample(dataset)
    dataset=balanced_ds.shuffle(1000)

    if augment:
        dataset=augment_data(dataset)
    dataset=dataset.batch(32)
    dataset=dataset.prefetch(tf.data.AUTOTUNE)

    logger.debug("preprocessing completed.")
    return dataset