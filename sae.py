
import time
import numpy as np
import pandas as pd
import tensorflow
from imblearn.over_sampling import SMOTE
from C45 import C45Classifier

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score ,roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input, BatchNormalization, Lambda
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers,regularizers

from matplotlib import pyplot



def train_orig_svc(x_train, x_test, y_train, y_test):
  model = SVC()
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)

  acc = accuracy_score(y_test, y_pred)
  auc = roc_auc_score(y_test, y_pred, multi_class="ovo")

  return acc, auc

def train_orig_knn(x_train, x_test, y_train, y_test) :
  knn = KNeighborsClassifier()
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test)

  #標籤轉換
  # y_pred = label_binarize(y_pred, classes=yy)
  # y_test = label_binarize(y_test, classes=yy)

  #分數
  acc = accuracy_score(y_test, y_pred)
  auc = roc_auc_score(y_test, y_pred, multi_class="ovo")

  return acc, auc


def transfer_y(y) :
  return np.unique(y) # 多分類AUC轉換使用


def data_preprocess(df_train, df_test) :

  x_train = df_train.drop(['Class'], axis=1)
  x_test = df_test.drop(['Class'], axis=1)

  # 特徵縮放
  minmax = preprocessing.MinMaxScaler()
  x_train_minmax = minmax.fit_transform(x_train)
  x_test_minmax = minmax.fit_transform(x_test)

  x_train = pd.DataFrame(x_train_minmax, columns = x_train.columns)
  x_test = pd.DataFrame(x_test_minmax, columns = x_test.columns)

  # Label encode
  labelencoder = LabelEncoder()
  y_train = labelencoder.fit_transform(df_train['Class'])
  y_test = labelencoder.fit_transform(df_test['Class'])

  return x_train, x_test, y_train, y_test

  # classifier

def run_svc(x_train, x_test, y_train, y_test):

  model = SVC(kernel='linear')
  model.fit(x_train, y_train)
  y_predict = model.predict(x_test)

  return roc_auc_score(y_test, y_predict, multi_class="ovo")


def run_knn(x_train, x_test, y_train, y_test):

  model = KNeighborsClassifier()
  model.fit(x_train, y_train)
  y_predict = model.predict(x_test)

  return roc_auc_score(y_test, y_predict, multi_class="ovo")


def run_c45(x_train, x_test, y_train, y_test):

  model = C45Classifier()
  model.fit(x_train, y_train)
  y_predict = model.predict(x_test)

  return roc_auc_score(y_test, y_predict)


def run_cart(x_train, x_test, y_train, y_test):
  
  model = tree.DecisionTreeClassifier()
  model.fit(x_train, y_train)
  y_predict = model.predict(x_test)

  return roc_auc_score(y_test, y_predict)

# SAE
from keras import regularizers

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
