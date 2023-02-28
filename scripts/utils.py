from tensorflow.keras.models import load_model  # type: ignore
import numpy as np


class Setup:
    """Loads the model and required dependencies. Prepares the model for making predictions
    """
    # list of alphabets
    __class_names = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    def __init__(self) -> None:
        # load model
        self.__model = load_model("./models/sequential_300_100.h5")

    def predict(self, X: np.ndarray) -> list[str]:
        # normalize the array
        array = X / 255.0

        # make predictions
        predictions = self.__model.predict(array).round(2)
        # get prediction classes
        classes = predictions.argmax(axis=1)
        # get letters from classes
        letters = [self.__class_names[c] for c in classes]

        # return letters
        return letters
