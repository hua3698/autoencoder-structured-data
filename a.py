#現在跑DAE

import numpy as np
import pandas as pd
import tensorflow
from imblearn.over_sampling import SMOTE
from C45 import C45Classifier

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score ,roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input, BatchNormalization
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


  #分數
  acc = accuracy_score(y_test, y_pred)
  auc = roc_auc_score(y_test, y_pred, multi_class="ovo")
  return acc, auc

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

datasets = ['vehicle1', 'page-blocks0', 'wisconsin', 'yeast1', 'haberman']

version = 'sae_510'
result = pd.DataFrame()

for idx, dataset in enumerate(datasets):

  svm = []; knn = []; c45 = []; cart = []
  svm_ae_smote = []; knn_ae_smote = []; c45_ae_smote = []; cart_ae_smote = []
  svm_smote_ae = []; knn_smote_ae = []; c45_smote_ae = []; cart_smote_ae = []

  for times in range(1,6):

    training = "{}-5-{}{}.dat".format(dataset, times, 'tra')
    testing = "{}-5-{}{}.dat".format(dataset, times, 'tst')

    df_train = pd.read_csv('../dataset/' + dataset + '-5-fold' + '/' + training, delimiter=',')
    df_test = pd.read_csv('../dataset/' + dataset + '-5-fold' + '/' + testing, delimiter=',')

    x_train, x_test, y_train, y_test = data_preprocess(df_train, df_test)

    # ae
    x_train_encoded, x_test_encoded = train_sae_510(x_train, x_test)

    auc1 = run_svc(x_train_encoded, x_test_encoded, y_train, y_test)
    auc2 = run_knn(x_train_encoded, x_test_encoded, y_train, y_test)
    auc3 = run_c45(x_train_encoded, x_test_encoded, y_train, y_test)
    auc4 = run_cart(x_train_encoded, x_test_encoded, y_train, y_test)

    svm.append(auc1); knn.append(auc2); c45.append(auc3); cart.append(auc4)

    # ae+smote
    x_train_ae_smote, y_train_ae_smote = SMOTE(random_state=42).fit_resample(x_train_encoded, y_train)

    auc5 = run_svc(x_train_ae_smote, x_test_encoded, y_train_ae_smote, y_test)
    auc6 = run_knn(x_train_ae_smote, x_test_encoded, y_train_ae_smote, y_test)
    auc7 = run_c45(x_train_ae_smote, x_test_encoded, y_train_ae_smote, y_test)
    auc8 = run_cart(x_train_ae_smote, x_test_encoded, y_train_ae_smote, y_test)

    svm_ae_smote.append(auc5); knn_ae_smote.append(auc6); c45_ae_smote.append(auc7); cart_ae_smote.append(auc8)

    # smote+ae
    x_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(x_train, y_train)
    x_train_smote_encoded, x_test_smote_encoded = train_sae_510(x_train_smote, x_test)

    auc9 = run_svc(x_train_smote_encoded, x_test_smote_encoded, y_train_smote, y_test)
    auc10 = run_knn(x_train_smote_encoded, x_test_smote_encoded, y_train_smote, y_test)
    auc11 = run_c45(x_train_smote_encoded, x_test_smote_encoded, y_train_smote, y_test)
    auc12 = run_cart(x_train_smote_encoded, x_test_smote_encoded, y_train_smote, y_test)

    svm_smote_ae.append(auc9); knn_smote_ae.append(auc10); c45_smote_ae.append(auc11); cart_smote_ae.append(auc12)


  new=pd.DataFrame({'dataset':dataset,
            'svm': np.mean(svm),
            'svm_ae_smote': np.mean(svm_ae_smote),
            'svm_smote_ae': np.mean(svm_smote_ae),
            'knn': np.mean(knn),
            'knn_ae_smote': np.mean(knn_ae_smote),
            'knn_smote_ae': np.mean(knn_smote_ae),
            'c45': np.mean(c45),
            'c45_ae_smote': np.mean(c45_ae_smote),
            'c45_smote_ae': np.mean(c45_smote_ae),
            'cart': np.mean(cart),
            'cart_ae_smote': np.mean(cart_ae_smote),
            'cart_smote_ae': np.mean(cart_smote_ae),
            }, index=[idx])

  print(version)
  print(new)
  
  new.to_csv('result/(ans)'+ version + '_' + dataset + '.csv',index=False,encoding='utf-8')

version = 'sae_610'

result = pd.DataFrame()

for idx, dataset in enumerate(datasets):

  svm = []; knn = []; c45 = []; cart = []
  svm_ae_smote = []; knn_ae_smote = []; c45_ae_smote = []; cart_ae_smote = []
  svm_smote_ae = []; knn_smote_ae = []; c45_smote_ae = []; cart_smote_ae = []

  for times in range(1,6):

    training = "{}-5-{}{}.dat".format(dataset, times, 'tra')
    testing = "{}-5-{}{}.dat".format(dataset, times, 'tst')

    df_train = pd.read_csv('../dataset/' + dataset + '-5-fold' + '/' + training, delimiter=',')
    df_test = pd.read_csv('../dataset/' + dataset + '-5-fold' + '/' + testing, delimiter=',')

    x_train, x_test, y_train, y_test = data_preprocess(df_train, df_test)

    # ae
    x_train_encoded, x_test_encoded = train_sae_610(x_train, x_test)

    auc1 = run_svc(x_train_encoded, x_test_encoded, y_train, y_test)
    auc2 = run_knn(x_train_encoded, x_test_encoded, y_train, y_test)
    auc3 = run_c45(x_train_encoded, x_test_encoded, y_train, y_test)
    auc4 = run_cart(x_train_encoded, x_test_encoded, y_train, y_test)

    svm.append(auc1); knn.append(auc2); c45.append(auc3); cart.append(auc4)

    # ae+smote
    x_train_ae_smote, y_train_ae_smote = SMOTE(random_state=42).fit_resample(x_train_encoded, y_train)

    auc5 = run_svc(x_train_ae_smote, x_test_encoded, y_train_ae_smote, y_test)
    auc6 = run_knn(x_train_ae_smote, x_test_encoded, y_train_ae_smote, y_test)
    auc7 = run_c45(x_train_ae_smote, x_test_encoded, y_train_ae_smote, y_test)
    auc8 = run_cart(x_train_ae_smote, x_test_encoded, y_train_ae_smote, y_test)

    svm_ae_smote.append(auc5); knn_ae_smote.append(auc6); c45_ae_smote.append(auc7); cart_ae_smote.append(auc8)

    # smote+ae
    x_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(x_train, y_train)
    x_train_smote_encoded, x_test_smote_encoded = train_sae_610(x_train_smote, x_test)

    auc9 = run_svc(x_train_smote_encoded, x_test_smote_encoded, y_train_smote, y_test)
    auc10 = run_knn(x_train_smote_encoded, x_test_smote_encoded, y_train_smote, y_test)
    auc11 = run_c45(x_train_smote_encoded, x_test_smote_encoded, y_train_smote, y_test)
    auc12 = run_cart(x_train_smote_encoded, x_test_smote_encoded, y_train_smote, y_test)

    svm_smote_ae.append(auc9); knn_smote_ae.append(auc10); c45_smote_ae.append(auc11); cart_smote_ae.append(auc12)


  new=pd.DataFrame({'dataset':dataset,
            'svm': np.mean(svm),
            'svm_ae_smote': np.mean(svm_ae_smote),
            'svm_smote_ae': np.mean(svm_smote_ae),
            'knn': np.mean(knn),
            'knn_ae_smote': np.mean(knn_ae_smote),
            'knn_smote_ae': np.mean(knn_smote_ae),
            'c45': np.mean(c45),
            'c45_ae_smote': np.mean(c45_ae_smote),
            'c45_smote_ae': np.mean(c45_smote_ae),
            'cart': np.mean(cart),
            'cart_ae_smote': np.mean(cart_ae_smote),
            'cart_smote_ae': np.mean(cart_smote_ae),
            }, index=[idx])

  print(version)
  print(new)
  
  new.to_csv('result/(ans)'+ version + '_' + dataset + '.csv',index=False,encoding='utf-8')

version = 'sae_710'
result = pd.DataFrame()

for idx, dataset in enumerate(datasets):

  svm = []; knn = []; c45 = []; cart = []
  svm_ae_smote = []; knn_ae_smote = []; c45_ae_smote = []; cart_ae_smote = []
  svm_smote_ae = []; knn_smote_ae = []; c45_smote_ae = []; cart_smote_ae = []

  for times in range(1,6):

    training = "{}-5-{}{}.dat".format(dataset, times, 'tra')
    testing = "{}-5-{}{}.dat".format(dataset, times, 'tst')

    df_train = pd.read_csv('../dataset/' + dataset + '-5-fold' + '/' + training, delimiter=',')
    df_test = pd.read_csv('../dataset/' + dataset + '-5-fold' + '/' + testing, delimiter=',')

    x_train, x_test, y_train, y_test = data_preprocess(df_train, df_test)

    # ae
    x_train_encoded, x_test_encoded = train_sae_710(x_train, x_test)

    auc1 = run_svc(x_train_encoded, x_test_encoded, y_train, y_test)
    auc2 = run_knn(x_train_encoded, x_test_encoded, y_train, y_test)
    auc3 = run_c45(x_train_encoded, x_test_encoded, y_train, y_test)
    auc4 = run_cart(x_train_encoded, x_test_encoded, y_train, y_test)

    svm.append(auc1); knn.append(auc2); c45.append(auc3); cart.append(auc4)

    # ae+smote
    x_train_ae_smote, y_train_ae_smote = SMOTE(random_state=42).fit_resample(x_train_encoded, y_train)

    auc5 = run_svc(x_train_ae_smote, x_test_encoded, y_train_ae_smote, y_test)
    auc6 = run_knn(x_train_ae_smote, x_test_encoded, y_train_ae_smote, y_test)
    auc7 = run_c45(x_train_ae_smote, x_test_encoded, y_train_ae_smote, y_test)
    auc8 = run_cart(x_train_ae_smote, x_test_encoded, y_train_ae_smote, y_test)

    svm_ae_smote.append(auc5); knn_ae_smote.append(auc6); c45_ae_smote.append(auc7); cart_ae_smote.append(auc8)

    # smote+ae
    x_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(x_train, y_train)
    x_train_smote_encoded, x_test_smote_encoded = train_sae_710(x_train_smote, x_test)

    auc9 = run_svc(x_train_smote_encoded, x_test_smote_encoded, y_train_smote, y_test)
    auc10 = run_knn(x_train_smote_encoded, x_test_smote_encoded, y_train_smote, y_test)
    auc11 = run_c45(x_train_smote_encoded, x_test_smote_encoded, y_train_smote, y_test)
    auc12 = run_cart(x_train_smote_encoded, x_test_smote_encoded, y_train_smote, y_test)

    svm_smote_ae.append(auc9); knn_smote_ae.append(auc10); c45_smote_ae.append(auc11); cart_smote_ae.append(auc12)


  new=pd.DataFrame({'dataset':dataset,
            'svm': np.mean(svm),
            'svm_ae_smote': np.mean(svm_ae_smote),
            'svm_smote_ae': np.mean(svm_smote_ae),
            'knn': np.mean(knn),
            'knn_ae_smote': np.mean(knn_ae_smote),
            'knn_smote_ae': np.mean(knn_smote_ae),
            'c45': np.mean(c45),
            'c45_ae_smote': np.mean(c45_ae_smote),
            'c45_smote_ae': np.mean(c45_smote_ae),
            'cart': np.mean(cart),
            'cart_ae_smote': np.mean(cart_ae_smote),
            'cart_smote_ae': np.mean(cart_smote_ae),
            }, index=[idx])

  print(version)
  print(new)
  
  new.to_csv('result/(ans)'+ version + '_' + dataset + '.csv',index=False,encoding='utf-8')

