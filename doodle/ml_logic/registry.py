import glob
import os
import time
import pickle

import tensorflow as tf
from colorama import Fore, Style
import keras


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    local_registry_path = os.path.join(os.getcwd(), 'model')
    
    # Save model locally
    model_path = os.path.join(local_registry_path, f"{timestamp}.keras")
    print(model_path)
    # breakpoint()
    model.save(model_path)

    print("✅ Model saved locally")

    return None
