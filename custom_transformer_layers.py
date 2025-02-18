import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow as tf

@keras.saving.register_keras_serializable()
class EmbeddingBlock(layers.Layer):
    def __init__(self, embedding_dim=64, **kwargs):
        super(EmbeddingBlock, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.dense = layers.Dense(embedding_dim, activation='relu')
        self.layer_norm = layers.LayerNormalization()
        self.positional_encoding = PositionalEncodingLayer(embedding_dim)

    def call(self, inputs):
        # Apply dense layer and layer normalization
        x = self.dense(inputs)
        x = self.layer_norm(x)
        # Apply positional encoding
        x = self.positional_encoding(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            "embedding_dim": self.embedding_dim,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.saving.register_keras_serializable()
class PositionalEncodingLayer(layers.Layer):
    def __init__(self, embedding_dim=64, **kwargs):
        super(PositionalEncodingLayer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def call(self, inputs):
        # Encode the relative positions of tokens in the input sequence 
        # by using a combination of sine and cosine functions with different frequencies  
        # so the model can process sequences of varying lengths effectively.
        # Details in 3.5 - Positional Encoding of the paper Attention is All You Need (Vaswani et al., 2017)

        # Generate positional encodings
        positions = tf.range(tf.shape(inputs)[1], dtype=tf.float32)
        positions = tf.expand_dims(positions, axis=-1)
        angles = tf.range(self.embedding_dim, dtype=tf.float32)
        angles = 1 / tf.pow(10000.0, (2 * (angles // 2)) / self.embedding_dim)
        angles = tf.expand_dims(angles, axis=0)
        positional_encodings = tf.matmul(positions, angles)
        positional_encodings = tf.concat(
            [tf.sin(positional_encodings[:, ::2]), tf.cos(positional_encodings[:, 1::2])], axis=-1
        )
        positional_encodings = tf.expand_dims(positional_encodings, axis=1)
        positional_encodings = tf.tile(positional_encodings, [tf.shape(inputs)[0], 1, 1])
        positional_encodings = tf.reshape(positional_encodings, tf.shape(inputs))
        # Add positional encodings to the input
        return inputs + positional_encodings

    def get_config(self):
        base_config = super().get_config()
        config = {
            "embedding_dim": self.embedding_dim,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.saving.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embedding_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embedding_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        # Multi-head self-attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)