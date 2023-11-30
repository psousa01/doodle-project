import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style

from doodle.ml_logic.model import *
from doodle.ml_logic.data import get_data, preprocess
from doodle.ml_logic.registry import save_model
from doodle.params import *


def train(
        learning_rate=0.0005,
        epochs = 100,
        patience = 5
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading data..." + Style.RESET_ALL)

    train_ds, val_ds = get_data(DATA_SIZE)
    
    # X_train, y_train = preprocess(train_ds)
    # X_val, y_val = preprocess(val_ds)
    # train_ds, val_ds = preprocess(train_ds, val_ds)
    
    # Train model using `model.py`
    model = initialize_model()
    model = compile_model(model=model, learning_rate=learning_rate)


    model, history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds = val_ds,
        epochs=epochs,
        patience=patience
    )

    val_accuracy = np.min(history.history['val_accuracy'])

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")

    return val_accuracy