import sqlite3
from typing import List
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
from utils import dict_factory


class MLBPredictor:

    def __init__(self, db_location: str = "mlb.db") -> None:
        self.conn = sqlite3.connect("mlb.db")
        self.conn.row_factory = dict_factory
        self.cursor = self.conn.cursor()
        self.x = list()
        self.y = list()
        self.model = None

    def train(
        self,
        start_year: int = 2000,
        end_year: int = 2023,
        features: List[str] = ["ERA", "R"],
        save_to: str = "mlb.keras",
    ):
        for i in range(start_year, end_year):
            self.cursor.execute(f"SELECT * FROM elo WHERE season = {i}")
            results = self.cursor.fetchall()
            for result in results:
                self.cursor.execute(
                    f"SELECT * FROM teams WHERE yearID = {i} and franchID = '{result['team1']}'"
                )
                res1 = self.cursor.fetchone()
                self.cursor.execute(
                    f"SELECT * FROM teams WHERE yearID = {i} and franchID = '{result['team2']}'"
                )
                res2 = self.cursor.fetchone()
                self.x.append(
                    [*[res1[f] for f in features], *[res2[f] for f in features]]
                )
                self.y.append(1 if result["score1"] > result["score2"] else 0)
                print(
                    result["team1"],
                    result["team2"],
                    res1["ERA"],
                    res1["R"],
                    res2["ERA"],
                    res1["R"],
                    1 if result["score1"] > result["score2"] else 0,
                )

        x = np.asarray(self.x).astype("float32")
        y = np.asarray(self.y).astype("float32")

        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        shape = len(features) * 2
        model = Sequential(
            [
                Dense(64, activation="relu", input_shape=(shape,)),
                Dense(32, activation="relu"),
                Dense(
                    1, activation="sigmoid"
                ),  # Output layer with sigmoid activation for binary classification
            ]
        )

        # Compile the model
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

        # Evaluate the model on the test set
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")

        model.save(save_to)
        self.model = model
        return model
    
    def load(self, model_path: str = "mlb.keras"):
        self.model = keras.models.load_model(model_path)
        return self.model
