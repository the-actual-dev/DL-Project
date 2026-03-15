"""Model architecture for Deep CNN-LSTM with Gated Self-Attention."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class GatedSelfAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Self-attention weights
        self.Wq = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="Wq",
        )
        self.Wk = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="Wk",
        )
        self.Wv = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="Wv",
        )

        # Gating layer
        self.gate_dense = layers.Dense(
            input_shape[-1],
            activation="sigmoid",
            name="attention_gate",
        )

        super().build(input_shape)

    def call(self, inputs):
        # ----- Self Attention -----
        Q = tf.matmul(inputs, self.Wq)
        K = tf.matmul(inputs, self.Wk)
        V = tf.matmul(inputs, self.Wv)

        scores = tf.matmul(Q, K, transpose_b=True)
        weights = tf.nn.softmax(scores, axis=-1)
        attention_output = tf.matmul(weights, V)

        # ----- Gating Mechanism -----
        gate = self.gate_dense(inputs)

        return attention_output * gate

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_m4_model(
    input_shape: tuple[int, int] = (10, 9),
    num_classes: int = 4,
) -> tf.keras.Model:
    """Build CNN-LSTM with Gated Self-Attention model."""

    inputs = layers.Input(shape=input_shape, name="input_sequence")

    # ----- CNN -----
    x = layers.Conv1D(
        filters=16,
        kernel_size=5,
        padding="valid",
        activation=None,
        name="conv1d_m4",
    )(inputs)
    x = layers.BatchNormalization(name="conv1d_bn")(x)
    x = layers.Dropout(0.2, name="conv1d_dropout")(x)

    # ----- LSTM + Gated Attention -----
    x = layers.LSTM(64, return_sequences=True, name="lstm_1")(x)
    x = GatedSelfAttention(64, name="gated_attention_1")(x)

    x = layers.LSTM(64, return_sequences=True, name="lstm_2")(x)
    x = GatedSelfAttention(64, name="gated_attention_2")(x)

    x = layers.LSTM(128, return_sequences=False, name="lstm_3")(x)

    # Projection
    x = layers.Dense(384, activation=None, name="projection_384")(x)

    # ----- Branch A -----
    branch_a = layers.Dense(320, name="branch_a_dense")(x)
    branch_a = layers.BatchNormalization(name="branch_a_bn")(branch_a)
    branch_a = layers.ReLU(name="branch_a_relu")(branch_a)
    branch_a = layers.Dropout(0.2, name="branch_a_dropout")(branch_a)

    # ----- Branch B -----
    branch_b = layers.Dense(512, name="branch_b_dense1")(x)
    branch_b = layers.ReLU(name="branch_b_relu1")(branch_b)
    branch_b = layers.Dense(64, name="branch_b_dense2")(branch_b)
    branch_b = layers.BatchNormalization(name="branch_b_bn2")(branch_b)
    branch_b = layers.ReLU(name="branch_b_relu2")(branch_b)
    branch_b = layers.Dropout(0.2, name="branch_b_dropout")(branch_b)

    merged = layers.Concatenate(name="concat_branches")([branch_a, branch_b])

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classifier",
    )(merged)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="M4_CNN_LSTM_GatedAttention",
    )
