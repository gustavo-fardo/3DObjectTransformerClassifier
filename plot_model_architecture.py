import keras
import numpy as np
import tensorflow as tf
from keras import layers, models
import tensorflow as tf
from keras.utils import plot_model

# Network input dimensions
num_classes = 10
num_points = 3*1024
input_shape = (num_points, 3)  # Shape of the input layer (num_points, num_axis)
embedding_dim = 128             # Expands the feature space from num_axis to embedding_dim (num_points, embedding_dim)
rate=0.1

# Transformer block size (total number of heads will be num_heads * num_layers)
num_layers = 2                # Number of transformer blocks
num_heads = 4                  # Number of attention heads

# Feed forward network size
ff_dim = 128                   # Hidden layer size in feed forward network inside transformer

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

    def plot_model(self, file_path='embedding_block_model.png'):
        # Create a temporary model to visualize the TransformerBlock
        inputs = keras.Input(shape=(None, self.embedding_dim))
        outputs = self.call(inputs)
        model = keras.Model(inputs, outputs)
        keras.utils.plot_model(model, to_file=file_path, show_shapes=True, show_layer_names=False)

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

    def plot_model(self, file_path='transformer_block_model.png'):
        # Create a temporary model to visualize the TransformerBlock
        inputs = keras.Input(shape=(None, self.embedding_dim))
        outputs = self.call(inputs)
        model = keras.Model(inputs, outputs)
        keras.utils.plot_model(model, to_file=file_path, show_shapes=True, show_layer_names=False)

inputs = layers.Input(shape=input_shape)

# Embedding block
x = layers.Dense(embedding_dim, activation='relu')(inputs)
x = layers.LayerNormalization()(x)
x = PositionalEncodingLayer(embedding_dim)(x)

# 2 Transformer blocks

# Multi-head self-attention
attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x)
attn_output = layers.Dropout(rate)(attn_output)
out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
# Feed-forward network
ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
ffn_output = layers.Dense(embedding_dim)(ffn_output)
ffn_output = layers.Dropout(rate)(ffn_output)
x = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

# Multi-head self-attention
attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x)
attn_output = layers.Dropout(rate)(attn_output)
out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
# Feed-forward network
ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
ffn_output = layers.Dense(embedding_dim)(ffn_output)
ffn_output = layers.Dropout(rate)(ffn_output)
x = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

# Global pooling and output
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)
visualkeras.layered_view(model, draw_volume=True, to_file='output.png', spacing=10, legend=True, scale_xy=1.5, max_xy=500, one_dim_orientation='x', show_dimension=True) # write to disk

plot_model(model, to_file='complete_model.png', 
            show_shapes=True,
            show_dtype=False,
            show_layer_names=False,
            rankdir="TB",
            expand_nested=False,
            dpi=200,
            show_trainable=False
)

tb = TransformerBlock(embedding_dim, num_heads, ff_dim)
tb.plot_model()

eb = EmbeddingBlock(embedding_dim)
eb.plot_model()

def build_transformer_model(input_shape, num_classes, embedding_dim=64, num_heads=4, ff_dim=128, num_layers=2):
    inputs = layers.Input(shape=input_shape)

    # Embedding block
    embedding_layer = EmbeddingBlock(embedding_dim=embedding_dim)
    x = embedding_layer(inputs)

    # Transformer blocks
    transformer_block = TransformerBlock(embedding_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = transformer_block(x)

    # Global pooling and output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

model = build_transformer_model(input_shape, num_classes, embedding_dim, num_heads, ff_dim, num_layers)

plot_model(model, to_file='short_plot_model.png', 
            show_shapes=True,
            show_dtype=False,
            show_layer_names=False,
            rankdir="TB",
            expand_nested=False,
            dpi=200,
            show_trainable=False
)
