import pandas as pd
import sklearn.model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import keras
import numpy as np

model_NN_weights = "/home/avinoam/workspace/YAM_HAMELACH/results/17_05/weights/cp.weights.h5"
data ="/home/avinoam/workspace/YAM_HAMELACH/results/17_05/features_extract/features_20240502_1525.csv"

model_NN = keras.models.Sequential( (
    keras.layers.InputLayer( (7,) ),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(2, activation=keras.activations.softmax)
)
)
if __name__ == "__main__":
    model_NN.compile(
        optimizer=keras.optimizers.Adam(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.CategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.categorical_accuracy],
    )


    df = pd.read_csv(data)
    features_columns = [ 'n_good', 'error_0.0-1.0', 'error_1.0-4.0',
           'error_4.0-16.0', 'error_16.0-64.0', 'error_64.0-256.0',
           'error_256.0-4096.0']

    dumies = pd.get_dummies(df, columns=["label"])

    X = df[features_columns]
    Y = dumies[["label_False_pair","label_True_pair"]].values
    Y = Y.astype(int)


    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, Y, test_size=0.4, random_state=42
        )


    history = model_NN.fit(
        X_train,
        y_train,
        batch_size=4,
        epochs=100,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_test, y_test),
    )

'''
X = df[features_columns]
Y = df['label']=="True_pair"
Y = Y.astype(int)

# model = SVC()
# model = KNeighborsClassifier()
model = DecisionTreeClassifier(max_depth=5, random_state=42)
# model = GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)
model.fit(X_train,y_train)

pred = model.predict(X_test)
print ((pred==y_test).mean())

# X_true = df[df.label=="True_pair"][features_columns]
# X_false = df[df.label!="True_pair"][features_columns]
# print((model.predict(X_true)==1))
# print((model.predict(X_false)==0))

pass
'''