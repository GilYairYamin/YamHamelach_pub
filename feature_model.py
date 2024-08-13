import pandas as pd
import sklearn.model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import optimizers

import tensorflow as tf
import keras
import numpy as np

data_df = pd.read_csv(data)
n_features = data_df.shape[1]-3

model_NN = keras.models.Sequential( (
    keras.layers.InputLayer( (n_features,) ),
    keras.layers.Normalization(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    # keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(32, activation='relu'),

    keras.layers.Dense(2, activation=keras.activations.softmax)
)
)

if __name__ == "__main__":

    # model_NN.load_weights(model_NN_weights)
    model_NN.compile(
        optimizer=optimizers.Adam(learning_rate=0.0002),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.CategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.categorical_accuracy],
    )


    df = pd.read_csv(data)
    features_columns = df.columns[:-3]
    valid = np.isnan(df[features_columns]).any(axis=1)==False
    print(f"data set shape is {df.shape}")
    df = df[valid]
    print(f"valid data set shape is {df.shape}")

    dumies = pd.get_dummies(df, columns=["label"])
    X = df[features_columns]
    Y = dumies[["label_False","label_True"]].values
    Y = Y.astype(int)


    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, Y, test_size=0.3, random_state=42
        )


    history = model_NN.fit(
        X_train,
        y_train,
        batch_size=8,
        epochs=100,

        validation_data=(X_test, y_test),
    )
    model_NN.save_weights(model_NN_weights)