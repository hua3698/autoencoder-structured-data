
from common import *
from library import *

# SAE
from keras import regularizers

def train_sae_210(x_train, x_test):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.9), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(int(input_dim*0.8), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=16,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_sae_220(x_train, x_test):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.8), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(int(input_dim*0.6), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=16,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_sae_230(x_train, x_test):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.7), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(int(input_dim*0.4), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=16,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_sae_240(x_train, x_test):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.6), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(int(input_dim*0.2), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=16,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_sae_h310(x_train, x_test):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.9), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(int(input_dim*0.7), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=16,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_sae_h320(x_train, x_test):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.8), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(int(input_dim*0.4), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=16,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_sae_h330(x_train, x_test):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.7), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.4), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(int(input_dim*0.1), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.4), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=16,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_sae_h410(x_train, x_test):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.9), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(int(input_dim*0.6), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=16,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_sae_420(x_train, x_test):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.8), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.4), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(int(input_dim*0.2), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.4), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=16,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_sae_510(x_train, x_test):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.9), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(int(input_dim*0.5), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=16,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_sae_610(x_train, x_test):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.9), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.5), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(int(input_dim*0.4), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.5), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=16,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded

def train_sae_710(x_train, x_test):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.9), activation='relu',activity_regularizer=regularizers.l1(10e-5))(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.8), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.7), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.6), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.5), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded = Dense(int(input_dim*0.4), activation='relu')(encoded)

  encoded = BatchNormalization()(encoded)
  encoded_end = Dense(int(input_dim*0.3), activation='relu')(encoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.4), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.5), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.6), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.7), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.8), activation='relu')(decoded)

  decoded = BatchNormalization()(encoded_end)
  decoded = Dense(int(input_dim*0.9), activation='relu')(decoded)

  decoded = BatchNormalization()(decoded)
  output_layer = Dense(input_dim, activation='sigmoid')(decoded)

  # 訓練
  autoencoder = Model(input_layer, output_layer)
  autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
  autoencoder.fit(x_train, x_train,
          epochs=100,
          batch_size=16,
          shuffle=True,
          validation_data=(x_test, x_test))

  encoder_model = Model(inputs=autoencoder.input, outputs=encoded_end)
  x_train_encoded = encoder_model.predict(x_train)
  x_test_encoded = encoder_model.predict(x_test)

  return x_train_encoded,x_test_encoded
