import tensorflow as tf
from tensorflow.keras.layers import Dropout,Dense
from tensorflow.keras.applications import Xception,ResNet50,MobileNetV2
from log import get_logger
import os
from data_ingestion import load_data
from preprocessing import preprocess

NUM_CLASSES = 4
EPOCHS = 15
LEARNING_RATE = 0.0001


logger=get_logger("Model_training")
artifacts_dir="artifacts"
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

def mobilenet_model(num_classes):
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
    os.makedirs(model_dir,exist_ok=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model"),
            monitor="val_loss",
            save_best_only=True,
            save_format="tf"
        )
    ]
    history=model.fit(train_ds,validation_data=val_ds,callbacks=callbacks,epochs=10)
    val_loss,val_acc=model.evaluate(val_ds)
    logger.debug(f"{model_name} Validation Accuracy: {val_acc:.4f}")

    logger.debug(f"Saved best {model_name} model to {model_dir}")

    return model

MODELS={
    "Xception":xception_model,
    "ResNet50":resNet_model,
    "MobileNetv2":mobilenet_model,
    "custom_cnn":custom_cnn}

if __name__=="__main__":

    raw_train_ds,raw_val_ds=load_data()
    
    for model_name, model_fn in MODELS.items():
        logger.info(f"Training {model_name}")
        train_ds=preprocess(dataset=raw_train_ds,model_name=model_name,augment=True)
        val_ds=preprocess(dataset=raw_val_ds,model_name=model_name,augment=False)
        model = model_fn(NUM_CLASSES)
        train_model( train_ds, val_ds,model,model_name)


