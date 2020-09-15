import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout, LayerNormalization, Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class ScaledDotProductAttention(Layer):
    def __init__(self, dk):
        super(ScaledDotProductAttention, self).__init__()
        self.dk = dk

    def call(self, query, key, value, mask=None):
        malmul_qk = tf.matmul(query, key, transpose_b=True)
        scale = tf.sqrt(tf.cast(self.dk, dtype=tf.float32))
        scores = malmul_qk / scale

        if mask is not None:
            scores += (mask * -1e9)

        scores = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(scores, value)
        return out, scores


class MultiHeadAttention(Layer):
    def __init__(self, head_count, d_model, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % head_count == 0

        self.head_count = head_count
        self.d_model = d_model
        self.dk = d_model // head_count

        # Chúng ta sẽ xây dựng 3 ma trận trọng số
        self.value = Dense(d_model)
        self.query = Dense(d_model)
        self.key = Dense(d_model)

        self.scaled = ScaledDotProductAttention(self.dk)
        self.dropout = Dropout(dropout)

        self.out = Dense(d_model)

    def call(self, v, k, q, mask=None):
        bs = tf.shape(q)[0] # batch_size

        q = self.query(q) # batch_size x seq_len X d_model
        k = self.key(k)
        v = self.value(v)

        q = tf.transpose(tf.reshape(q, (bs, -1, self.head_count, self.dk)), perm=[0, 1, 2, 3])
        k = tf.transpose(tf.reshape(k, (bs, -1, self.head_count, self.dk)), perm=[0, 1, 2, 3])
        v = tf.transpose(tf.reshape(v, (bs, -1, self.head_count, self.dk)), perm=[0, 1, 2, 3])

        out, scores = self.scaled(q, k, v, mask)
        out = tf.reshape(tf.transpose(out, perm=[0, 1, 2, 3]), (bs, -1, self.head_count, self.dk))
        out = self.out(out)
        return out, scores


class PositionwiseFeedForward(Layer):
    def __init__(self, d_model, dff):
        super(PositionwiseFeedForward, self).__init__()
        self.dropout = Dropout(0.1)
        self.w1 = Dense(d_model)
        self.w2 = Dense(dff)

    def call(self, x):
        out = self.w2(tf.nn.relu(self.w1(x)))
        return out


class Embedding(Layer):
    def __init__(self, d_model, vocab_size):
        super(Embedding, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.d_model = d_model

    def call(self, x):
        out = self.embedding(x) * tf.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        return out

class PositionalEncoding(Layer):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(0.1)
        self.d_model = d_model

        pos = np.expand_dims(np.arange(0, max_len), axis=1)
        index = np.expand_dims(np.arange(0, self.d_model), axis=0)
        pe = self.get_angles(pos, index)
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        self.pe = tf.cast(np.expand_dims(pe, axis=0), tf.float32)

    def call(self, x):
        x += self.pe
        out = self.dropout(x)
        return out

    def get_angles(self, pos, index):
        angle = pos / np.power(10000, (2*(index//2)/np.float32(self.d_model)))
        return angle


class EncoderLayer(Layer):
    def __init__(self, d_model, dff, head_count, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.head_count = head_count
        self.d_model = d_model
        self.dff = dff

        self.multi_head_attention = MultiHeadAttention(head_count, d_model, dropout)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.pwll = PositionwiseFeedForward(d_model, dff)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, x, mask):
        out, attention = self.multi_head_attention(x, x, x, mask)
        out = self.dropout1(out)
        out += x
        out = self.norm1(out)

        out = self.pwll(out)
        out = self.dropout2(out)
        out += x
        out = self.norm2(out)

        return out, attention


class Encoder(Layer):
    def __init__(self, vocab_size, d_model, num_layer, dff, head_count, dropout):
        super(Encoder, self).__init__()
        self.num_layer = num_layer
        self.embedding = Embedding(d_model, vocab_size)
        self.layers = [EncoderLayer(d_model, dff, head_count, dropout)  for _ in range(num_layer)]
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.norm = LayerNormalization(d_model)

    def call(self, input, mask):
        x = self.embedding(input)
        x = self.pe(x)
        for i in range(self.num_layer):
            x = self.layers[i](x, mask)
        out = self.norm(x)
        return out


class DecoderLayer(Layer):
    def __init__(self, head_count, d_model, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(head_count, d_model, dropout)
        self.attention2 = MultiHeadAttention(head_count, d_model, dropout)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)

        self.ff = PositionwiseFeedForward(d_model, dff)

    def call(self, input, encoder_out, ahead_mask, padding_mask):
        x = input
        x1 = self.norm1(input)
        x1, attention1 = self.attention1(x1, x1, x1, padding_mask)
        x += self.dropout1(x1)

        x2 = self.norm2(x)
        x2, attention2 = self.attention2(x2, encoder_out, encoder_out, ahead_mask)
        x += self.dropout2(x2)

        x3 = self.norm3(x)
        out = x + self.dropout3(x3)

        return out, attention1, attention2


class Decoder(Layer):
    def __init__(self, vocab_size, d_model, num_layer, dropout, dff):
        super(Decoder, self).__init__()
        self.num_layer = num_layer
        self.embedding = Embedding(d_model, vocab_size)
        self.pe = PositionalEncoding(d_model, dropout=dropout)



















