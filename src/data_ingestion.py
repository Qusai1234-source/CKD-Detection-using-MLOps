import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from log import get_logger
from tensorflow.keras.preprocessing import image_dataset_from_directory


logger=get_logger("data_Ingestion")

def validate_data(file_path):
    logger.debug("Validating data Path")

    if not os.path.exists(file_path):
        logger.debug("Dataset Directory not Found.")
        raise FileNotFoundError("Dataset Directory does not exist")
    
    classes=os.listdir(file_path)
    if(len(classes)==0):
        logger.debug("Classes does not exist in subdirectory.")
        raise Exception("Dataset has no class folder")
    for cls in classes:
        cls_path=os.path.join(file_path,cls)
        if(len(os.listdir(cls_path))==0):
            logger.debug("Images not found.")
            raise Exception("Dataset has no images.")
    
    logger.debug(f"Dataset Validation Successful. Classes found:{classes}")

def load_data(file_path=r"C:\projects\CKD-Detection-using-MLOps\Data\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"):

    logger.debug("Starting Image Data Ingestion.")

    train_ds=image_dataset_from_directory(
        file_path,
        validation_split=0.2,
        seed=42,
        batch_size=32,
        image_size=(224,224),
        label_mode="int",
        subset="training"
    )

    val_ds=image_dataset_from_directory(
        file_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224,224),
        batch_size=32,
        label_mode="int"
    )

    logger.debug("Image Datasets created successfully.")
    return train_ds,val_ds

if __name__=="__main__":
    file_path=r"C:\projects\CKD-Detection-using-MLOps\Data\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
    validate_data(file_path=file_path)
    train_ds,val_ds=load_data(file_path)
    print(len(train_ds),len(val_ds))
