import os
import numpy as np
import tensorflow as tf
import pandas as pd
from src.model import SelfAttention

MODEL_PATH = "results/best_model.h5"

print("Loading model from:", MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"SelfAttention": SelfAttention},
)

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Pick any 10 consecutive rows
start_idx = 100
window = df.iloc[start_idx:start_idx + 10][
    ["ax", "ay", "az", "gx", "gy", "gz", "lax", "lay", "laz"]
].values

# Shape must be (1, 10, 9)
window = np.expand_dims(window, axis=0)

# Predict
probs = model.predict(window)
pred_class = int(np.argmax(probs, axis=1)[0])

print("\nPredicted class:", pred_class)
print("Class probabilities:", probs[0])
