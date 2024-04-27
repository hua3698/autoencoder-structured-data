
from common import *
from library import *
import tensorflow.keras.backend as K
from tensorflow.keras import losses

def train_vae_210(x_train, x_test):

  input_dim = x_train.shape[1]

  latent_dim = input_dim*0.8
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.9), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='binary_crossentropy')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=16,
      shuffle=True,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_vae_220(x_train, x_test):

  input_dim = x_train.shape[1]

  latent_dim = input_dim*0.6
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.8), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='binary_crossentropy')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=16,
      shuffle=True,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_vae_230(x_train, x_test):

  input_dim = x_train.shape[1]

  latent_dim = input_dim*0.4
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.7), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='binary_crossentropy')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=16,
      shuffle=True,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_vae_240(x_train, x_test):

  input_dim = x_train.shape[1]

  latent_dim = input_dim*0.2
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.6), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='binary_crossentropy')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=16,
      shuffle=True,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_vae_310(x_train, x_test):

  input_dim = x_train.shape[1]

  latent_dim = input_dim*0.7
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.9), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='binary_crossentropy')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=16,
      shuffle=True,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_vae_320(x_train, x_test):

  input_dim = x_train.shape[1]

  latent_dim = input_dim*0.4
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='binary_crossentropy')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=16,
      shuffle=True,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_vae_330(x_train, x_test):

  input_dim = x_train.shape[1]

  latent_dim = input_dim*0.1
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.4), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.4), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='binary_crossentropy')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=16,
      shuffle=True,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_vae_410(x_train, x_test):

  input_dim = x_train.shape[1]

  latent_dim = input_dim*0.6
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.9), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='binary_crossentropy')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=16,
      shuffle=True,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_vae_420(x_train, x_test):

  input_dim = x_train.shape[1]

  latent_dim = input_dim*0.2
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.4), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.4), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='binary_crossentropy')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=16,
      shuffle=True,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_vae_510(x_train, x_test):

  input_dim = x_train.shape[1]

  latent_dim = input_dim*0.5
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.9), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='binary_crossentropy')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=16,
      shuffle=True,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_vae_610(x_train, x_test):

  input_dim = x_train.shape[1]

  latent_dim = input_dim*0.4
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.9), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.5), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.5), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='binary_crossentropy')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=16,
      shuffle=True,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_vae_710(x_train, x_test):

  input_dim = x_train.shape[1]

  latent_dim = input_dim*0.3
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)

  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.9), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.5), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.4), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  z_mean = enc_mean(encoded)
  z_log_var = enc_log_var(encoded)
  z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.4), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.5), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(z)
  decoded = Dense(int(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  vae = Model(input_layer, output_layer)

  #loss
  reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  vae.compile(optimizer='adam', loss='binary_crossentropy')
  vae.fit(x_train, x_train,
      epochs=100,
      batch_size=16,
      shuffle=True,
      validation_data=(x_test, x_test))

  vae_encoder = Model(vae.input, z_mean)
  x_train_encoded = vae_encoder.predict(x_train)
  x_test_encoded = vae_encoder.predict(x_test)

  return x_train_encoded,x_test_encoded
