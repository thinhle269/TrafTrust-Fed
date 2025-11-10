
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

def build_mlp(input_dim=12, lr=1e-3, loss_name="mae"):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    opt = Adam(learning_rate=lr)
    if loss_name == "huber":
        loss = tf.keras.losses.Huber(delta=0.1)
    else:
        loss = "mae"
    model.compile(optimizer=opt, loss=loss)
    return model
