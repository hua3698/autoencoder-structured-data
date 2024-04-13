# chat寫得VAE

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, models

# 定義編碼器
def build_encoder(input_dim, latent_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(input_dim*0.7, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    return models.Model(inputs, [z_mean, z_log_var], name='encoder')

# 定義解碼器
def build_decoder(latent_dim, output_dim):
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(512, activation='relu')(inputs)
    outputs = layers.Dense(output_dim, activation='sigmoid')(x)
    return models.Model(inputs, outputs, name='decoder')

# 定義抽樣函數
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# 損失函數：重建損失 + KL 散度
def vae_loss(input_dim, z_mean, z_log_var, reconstructed):
    reconstruction_loss = input_dim * losses.binary_crossentropy(inputs, reconstructed)
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    return tf.reduce_mean(reconstruction_loss + kl_loss)

# 建立 VAE 模型
def build_vae(encoder, decoder):
    inputs = layers.Input(shape=(input_dim,))
    z_mean, z_log_var = encoder(inputs)
    z = layers.Lambda(sampling)([z_mean, z_log_var])
    reconstructed = decoder(z)
    vae = models.Model(inputs, reconstructed, name='vae')
    vae.add_loss(vae_loss(input_dim, z_mean, z_log_var, reconstructed))
    return vae

# 訓練 VAE
def train_vae(x_train, input_dim, latent_dim, epochs=50, batch_size=16):
    encoder = build_encoder(input_dim, latent_dim)
    decoder = build_decoder(latent_dim, input_dim)
    vae = build_vae(encoder, decoder)
    vae.compile(optimizer='adam')
    vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True)
    return vae, encoder, decoder

# 主程序
if __name__ == "__main__":
    # 虛擬數據
    x_train = np.random.randn(10000, 784)
    input_dim = x_train.shape[1]
    latent_dim = 64
    
    # 訓練 VAE
    trained_vae, trained_encoder, trained_decoder = train_vae(x_train, input_dim, latent_dim)
