import numpy as np
import time

from colorama import Fore, Style

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from keras import Model, layers, optimizers
from keras.callbacks import EarlyStopping
from keras.applications.vgg16 import VGG16, preprocess_input

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")



def initialize_model():
    #Initialize the VGG16 base model
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(64,64,3)
    )
    base_model.trainable = False # Sets the VGG16 weights to not trainable

    #Keras Functional API
    inputs = layers.Input(shape = (64,64,3))
    
    x = preprocess_input(inputs) # Use same preprocessing as VGG16 to match the model
    # x = layers.Rescaling(scale=1./127.5, offset=-1)(inputs)
    x = base_model(x)
    
    x = layers.Flatten()(x)

    # Our own dense layers from now on
    x = layers.Dense(600, activation = 'relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(500, activation = 'relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(400, activation = 'relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    pred = layers.Dense(8, activation = 'softmax')(x) # 5 IS FOR REDUCED DATASET, CHANGE TO 345 WHEN USING FULL
    
    
    model = Model(inputs = inputs, outputs = pred)
    

    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        train_ds,
        val_ds,
        epochs=15,
        patience=5,
    ):
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[es],
        verbose=1
    )

    print(f"✅ Model trained on  rows with max val accuracy: {round(np.max(history.history['val_accuracy']), 3)}")

    return model, history
