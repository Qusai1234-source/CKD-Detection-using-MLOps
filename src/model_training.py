import tensorflow as tf
from tensorflow.keras.layers import Dropout,Dense
from tensorflow.keras.applications import Xception,ResNet50,MobileNetV2
from log import get_logger
import os
logger=get_logger("Model_training")
artifacts_dir=r"C:\projects\CKD-Detection-using-MLOps\artifacts"
def xception_model(num_classes):
    base=Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )
    base.trainable=False

    x=tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x=Dense(256,activation="relu")(x)
    x=Dropout(0.5)(x)
    output=Dense(num_classes,activation="softmax")(x)

    return tf.keras.Model(base.input,output)

def resNet_model(num_classes):
    base=ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )
    base.trainable=False

    x=tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x=Dense(256,activation="relu")(x)
    output=Dense(num_classes,activation="softmax")(x)

    return tf.keras.Model(base.input,output)

def mobilnet_model(num_classes):
    base=MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )
    base.trainable=False

    x=tf.keras.layers.GlobalAveragePooling2D()(base.output)
    output=Dense(num_classes,activation="softmax")(x)

    return tf.keras.Model(base.input,output)

def custom_cnn(num_classes):
    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,3,activation="relu",input_shape=(224,224,3)),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64,3,activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128,3,activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes,activation="softmax")
    ])
    return model

def train_model(train_ds,val_ds,model,model_name):

    logger.debug(f"Training Model:{model_name}")
    model_dir=os.path.join(artifacts_dir,model_name)
    os.makedirs(model_dir,exist_pk=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy,val_accuracy"]
    )
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True),

        tf.keras.callbacks.ModelCheckpoint(

        )
    ]
    history=model.fit(train_ds,validation_data=val_ds,callbacks=callbacks,epochs=10)
    
