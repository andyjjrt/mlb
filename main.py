import sqlite3
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from utils import dict_factory

conn = sqlite3.connect("mlb.db")
conn.row_factory = dict_factory

cursor = conn.cursor()
cursor.execute("SELECT * FROM elo WHERE season = 2022")
results = cursor.fetchall()

x = list()
y = list()

for result in results:
    cursor.execute(
        f"SELECT * FROM teams WHERE yearID = 2022 and franchID = '{result['team1']}'"
    )
    res1 = cursor.fetchone()
    cursor.execute(
        f"SELECT * FROM teams WHERE yearID = 2022 and franchID = '{result['team2']}'"
    )
    res2 = cursor.fetchone()
    x.append([res1["ERA"], res1["R"], res2["ERA"], res1["R"]])
    y.append(1 if result["score1"] > result["score2"] else 0)
    print(
        result["team1"],
        result["team2"],
        res1["ERA"],
        res1["R"],
        res2["ERA"],
        res1["R"],
        1 if result["score1"] > result["score2"] else 0,
    )

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

x = np.array(x)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Define the model architecture
model = Sequential(
    [
        Dense(64, activation="relu", input_shape=(4,)),
        Dense(32, activation="relu"),
        Dense(
            1, activation="sigmoid"
        ),  # Output layer with sigmoid activation for binary classification
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
