import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score ,roc_auc_score
#前處理
from sklearn.preprocessing import LabelEncoder,label_binarize
from keras.utils import np_utils
from sklearn import preprocessing
#time
import time
#MLP juypter
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input,BatchNormalization,Lambda
#VAE
import tensorflow.keras.backend as K
from tensorflow.keras import losses

#vae(前測)(2-1)
fliename=['(03)sonar','(08)arrhythmia','(14)oh15.wc']
ansc45knn = pd.DataFrame()
for ff in fliename:

  print(ff)
  file=ff
  df = pd.read_csv('org/'+file+'.csv')
  labelencoder = LabelEncoder()
  df['Class'] = labelencoder.fit_transform(df['Class'])

  y = df['Class']
  x = df.drop(['Class'], axis=1)
  x_c = df.drop(['Class'], axis=1)#取得columns名稱
  yy=np.unique(y)#多分類AUC轉換使用

  #資料正規化
  minmax = preprocessing.MinMaxScaler()
  x = minmax.fit_transform(x)
  x=pd.DataFrame(x,columns = x_c.columns)

  #算維度
  input_dim = x.shape[1]
  y_train_class = np_utils.to_categorical(y)
  nb_classes = y_train_class.shape[1]


  #測試用，記得條回 100和20
  my_callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=100,monitor = 'loss',mode='min',min_delta=0.0001,restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
  ]
  # callback
  my_callbacks2 = [
      tf.keras.callbacks.EarlyStopping(patience=50,monitor = 'accuracy',mode='max',min_delta=0.0001,restore_best_weights=True)

  ]


  def sampling(args):
      z_mean, z_log_var = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      # by default, random_normal has mean = 0 and std = 1.0
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + K.exp(0.5 * z_log_var) * epsilon


  latent_dim = input_dim*0.8
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)


  # AE+SVM+KNN5折交叉驗證

  sfolder = StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y)

  svm_acc = []; svm_auc = []; svm_time = []

  for k, (train, test) in enumerate(sfolder):

      #重新命名 Kfold的資料 .iloc
      kfold_x_train = x.iloc[train]
      kfold_y_train = y.iloc[train]
      kfold_x_test =  x.iloc[test]
      kfold_y_test =  y.iloc[test]

      #特徵選取時間開始
      start1 = time.process_time()

      #特徵選取
      input_layer = Input(shape = (input_dim, ))
      #e1
      encoded = BatchNormalization()(input_layer)
      encoded = Dense(input_dim*0.9, activation='relu')(encoded)
      #e2
      encoded = BatchNormalization()(encoded)
      z_mean = enc_mean(encoded)
      z_log_var = enc_log_var(encoded)
      z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

      #D2
      decoded = BatchNormalization()(z)
      decoded = Dense(input_dim*0.9, activation='relu')(decoded)

      #output
      decoded = BatchNormalization()(decoded)
      output_layer = Dense(input_dim, activation='sigmoid')(decoded)

      #model
      VAE = Model(input_layer, output_layer)
      #softmax
      Soft_output = Dense(nb_classes, activation='softmax')(z_mean)
      ae_soft = Model(input_layer, Soft_output)

      #loss
      reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)

      kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
      kl_loss = K.sum(kl_loss, axis=-1)
      kl_loss *= -0.5

      vae_loss = K.mean(reconstruction_loss + kl_loss)

      VAE.add_loss(vae_loss)

      #compile
      VAE.compile(optimizer='adam', loss='mse')
      VAE.fit(kfold_x_train, kfold_x_train,
                  epochs=1000,
                  batch_size=16,
                  shuffle=True,
                  callbacks=my_callbacks,
                  le(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      ae_soft.fit(kfold_x_train, kfold_y_train,
              batch_size=16,
              epochs=1000,
              shuffle=True,
              callbacks=my_callbacks2,
              validation_data=(kfold_x_test, kfold_y_test))

      y_predict = np.argmax(ae_soft.predict(kfold_x_test), axis=-1)

      #標籤轉換
      y_predict_m=label_binarize(y_predict, classes=yy)
      kfold_y_test_m=label_binarize(kfold_y_test, classes=yy)

      #分數
      acc = accuracy_score(kfold_y_test_m, y_predict_m)
      auc = roc_auc_score(kfold_y_test_m, y_predict_m,multi_class="ovo")

      #特徵選取時間結束
      end1 = time.process_time()
      tt1=end1 - start1

      #存各項數值
      svm_acc.append(acc); svm_auc.append(auc); svm_time.append(tt1)

  a=np.mean(svm_acc)
  b=np.mean(svm_auc)
  c=np.mean(svm_time)


  print(a)
  print(b)
  print(c)
  print("--------end--------")

  new=pd.DataFrame({'type':'(vae)(2-1)',
        'dataset':ff,
        'svm_acc':a,
        'svm_auc':b,
        'svm_time':c,
        },
                 index=[1])

  ansc45knn=ansc45knn.append(new)
  ansc45knn.to_csv('VAE/(ans)vae(前測)(2-1).csv',index=False,encoding='utf-8')

#vae(前測)(2-2)
fliename=['(03)sonar','(08)arrhythmia','(14)oh15.wc']
ansc45knn = pd.DataFrame()
for ff in fliename:

  print(ff)
  file=ff
  df = pd.read_csv('org/'+file+'.csv')
  labelencoder = LabelEncoder()
  df['Class'] = labelencoder.fit_transform(df['Class'])

  y = df['Class']
  x = df.drop(['Class'], axis=1)
  x_c = df.drop(['Class'], axis=1)#取得columns名稱
  yy=np.unique(y)#多分類AUC轉換使用

  #資料正規化
  minmax = preprocessing.MinMaxScaler()
  x = minmax.fit_transform(x)
  x=pd.DataFrame(x,columns = x_c.columns)

  #算維度
  input_dim = x.shape[1]
  y_train_class = np_utils.to_categorical(y)
  nb_classes = y_train_class.shape[1]


  #測試用，記得條回 100和20
  my_callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=100,monitor = 'loss',mode='min',min_delta=0.0001,restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
  ]
  # callback
  my_callbacks2 = [
      tf.keras.callbacks.EarlyStopping(patience=50,monitor = 'accuracy',mode='max',min_delta=0.0001,restore_best_weights=True)

  ]


  def sampling(args):
      z_mean, z_log_var = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      # by default, random_normal has mean = 0 and std = 1.0
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + K.exp(0.5 * z_log_var) * epsilon


  latent_dim = input_dim*0.6
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)


  # AE+SVM+KNN5折交叉驗證

  sfolder = StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y)

  svm_acc = []; svm_auc = []; svm_time = []

  for k, (train, test) in enumerate(sfolder):

      #重新命名 Kfold的資料 .iloc
      kfold_x_train = x.iloc[train]
      kfold_y_train = y.iloc[train]
      kfold_x_test =  x.iloc[test]
      kfold_y_test =  y.iloc[test]

      #特徵選取時間開始
      start1 = time.process_time()

      #特徵選取
      input_layer = Input(shape = (input_dim, ))
      #e1
      encoded = BatchNormalization()(input_layer)
      encoded = Dense(input_dim*0.8, activation='relu')(encoded)

      #e3
      encoded = BatchNormalization()(encoded)
      z_mean = enc_mean(encoded)
      z_log_var = enc_log_var(encoded)
      z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

      #D2
      decoded = BatchNormalization()(z)
      decoded = Dense(input_dim*0.8, activation='relu')(decoded)

      #output
      decoded = BatchNormalization()(decoded)
      output_layer = Dense(input_dim, activation='sigmoid')(decoded)

      #model
      VAE = Model(input_layer, output_layer)
      #softmax
      Soft_output = Dense(nb_classes, activation='softmax')(z_mean)
      ae_soft = Model(input_layer, Soft_output)

      #loss
      reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)

      kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
      kl_loss = K.sum(kl_loss, axis=-1)
      kl_loss *= -0.5

      vae_loss = K.mean(reconstruction_loss + kl_loss)

      VAE.add_loss(vae_loss)

      #compile
      VAE.compile(optimizer='adam', loss='mse')
      VAE.fit(kfold_x_train, kfold_x_train,
                  epochs=1000,
                  batch_size=16,
                  shuffle=True,
                  callbacks=my_callbacks,
                  )


      #softmax
      ae_soft.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      ae_soft.fit(kfold_x_train, kfold_y_train,
              batch_size=16,
              epochs=1000,
              shuffle=True,
              callbacks=my_callbacks2,
              validation_data=(kfold_x_test, kfold_y_test))

      y_predict = np.argmax(ae_soft.predict(kfold_x_test), axis=-1)

      #標籤轉換
      y_predict_m=label_binarize(y_predict, classes=yy)
      kfold_y_test_m=label_binarize(kfold_y_test, classes=yy)

      #分數
      acc = accuracy_score(kfold_y_test_m, y_predict_m)
      auc = roc_auc_score(kfold_y_test_m, y_predict_m,multi_class="ovo")

      #特徵選取時間結束
      end1 = time.process_time()
      tt1=end1 - start1

      #存各項數值
      svm_acc.append(acc); svm_auc.append(auc); svm_time.append(tt1)

  a=np.mean(svm_acc)
  b=np.mean(svm_auc)
  c=np.mean(svm_time)


  print(a)
  print(b)
  print(c)
  print("--------end--------")

  new=pd.DataFrame({'type':'(vae)(2-2)',
        'dataset':ff,
        'svm_acc':a,
        'svm_auc':b,
        'svm_time':c,
        },
                 index=[1])

  ansc45knn=ansc45knn.append(new)
  ansc45knn.to_csv('VAE/(ans)vae(前測)(2-2).csv',index=False,encoding='utf-8')

#vae(前測)(2-3)
fliename=['(03)sonar','(08)arrhythmia','(14)oh15.wc']
ansc45knn = pd.DataFrame()
for ff in fliename:

  print(ff)
  file=ff
  df = pd.read_csv('org/'+file+'.csv')
  labelencoder = LabelEncoder()
  df['Class'] = labelencoder.fit_transform(df['Class'])

  y = df['Class']
  x = df.drop(['Class'], axis=1)
  x_c = df.drop(['Class'], axis=1)#取得columns名稱
  yy=np.unique(y)#多分類AUC轉換使用

  #資料正規化
  minmax = preprocessing.MinMaxScaler()
  x = minmax.fit_transform(x)
  x=pd.DataFrame(x,columns = x_c.columns)

  #算維度
  input_dim = x.shape[1]
  y_train_class = np_utils.to_categorical(y)
  nb_classes = y_train_class.shape[1]


  #測試用，記得條回 100和20
  my_callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=100,monitor = 'loss',mode='min',min_delta=0.0001,restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
  ]
  # callback
  my_callbacks2 = [
      tf.keras.callbacks.EarlyStopping(patience=50,monitor = 'accuracy',mode='max',min_delta=0.0001,restore_best_weights=True)

  ]


  def sampling(args):
      z_mean, z_log_var = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      # by default, random_normal has mean = 0 and std = 1.0
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + K.exp(0.5 * z_log_var) * epsilon


  latent_dim = input_dim*0.4
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)


  # AE+SVM+KNN5折交叉驗證

  sfolder = StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y)

  svm_acc = []; svm_auc = []; svm_time = []

  for k, (train, test) in enumerate(sfolder):

      #重新命名 Kfold的資料 .iloc
      kfold_x_train = x.iloc[train]
      kfold_y_train = y.iloc[train]
      kfold_x_test =  x.iloc[test]
      kfold_y_test =  y.iloc[test]

      #特徵選取時間開始
      start1 = time.process_time()

      #特徵選取
      input_layer = Input(shape = (input_dim, ))
      #e1
      encoded = BatchNormalization()(input_layer)
      encoded = Dense(input_dim*0.7, activation='relu')(encoded)

      #e3
      encoded = BatchNormalization()(encoded)
      z_mean = enc_mean(encoded)
      z_log_var = enc_log_var(encoded)
      z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

      #D2
      decoded = BatchNormalization()(z)
      decoded = Dense(input_dim*0.7, activation='relu')(decoded)

      #output
      decoded = BatchNormalization()(decoded)
      output_layer = Dense(input_dim, activation='sigmoid')(decoded)

      #model
      VAE = Model(input_layer, output_layer)
      #softmax
      Soft_output = Dense(nb_classes, activation='softmax')(z_mean)
      ae_soft = Model(input_layer, Soft_output)

      #loss
      reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)

      kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
      kl_loss = K.sum(kl_loss, axis=-1)
      kl_loss *= -0.5

      vae_loss = K.mean(reconstruction_loss + kl_loss)

      VAE.add_loss(vae_loss)

      #compile
      VAE.compile(optimizer='adam', loss='mse')
      VAE.fit(kfold_x_train, kfold_x_train,
                  epochs=1000,
                  batch_size=16,
                  shuffle=True,
                  callbacks=my_callbacks,
                  )


      #softmax
      ae_soft.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      ae_soft.fit(kfold_x_train, kfold_y_train,
              batch_size=16,
              epochs=1000,
              shuffle=True,
              callbacks=my_callbacks2,
              validation_data=(kfold_x_test, kfold_y_test))

      y_predict = np.argmax(ae_soft.predict(kfold_x_test), axis=-1)

      #標籤轉換
      y_predict_m=label_binarize(y_predict, classes=yy)
      kfold_y_test_m=label_binarize(kfold_y_test, classes=yy)

      #分數
      acc = accuracy_score(kfold_y_test_m, y_predict_m)
      auc = roc_auc_score(kfold_y_test_m, y_predict_m,multi_class="ovo")

      #特徵選取時間結束
      end1 = time.process_time()
      tt1=end1 - start1

      #存各項數值
      svm_acc.append(acc); svm_auc.append(auc); svm_time.append(tt1)

  a=np.mean(svm_acc)
  b=np.mean(svm_auc)
  c=np.mean(svm_time)


  print(a)
  print(b)
  print(c)
  print("--------end--------")

  new=pd.DataFrame({'type':'(vae)(2-3)',
        'dataset':ff,
        'svm_acc':a,
        'svm_auc':b,
        'svm_time':c,
        },
                 index=[1])

  ansc45knn=ansc45knn.append(new)
  ansc45knn.to_csv('VAE/(ans)vae(前測)(2-3).csv',index=False,encoding='utf-8')

#vae(前測)(2-4)
fliename=['(03)sonar','(08)arrhythmia','(14)oh15.wc']
ansc45knn = pd.DataFrame()
for ff in fliename:

  print(ff)
  file=ff
  df = pd.read_csv('org/'+file+'.csv')
  labelencoder = LabelEncoder()
  df['Class'] = labelencoder.fit_transform(df['Class'])

  y = df['Class']
  x = df.drop(['Class'], axis=1)
  x_c = df.drop(['Class'], axis=1)#取得columns名稱
  yy=np.unique(y)#多分類AUC轉換使用

  #資料正規化
  minmax = preprocessing.MinMaxScaler()
  x = minmax.fit_transform(x)
  x=pd.DataFrame(x,columns = x_c.columns)

  #算維度
  input_dim = x.shape[1]
  y_train_class = np_utils.to_categorical(y)
  nb_classes = y_train_class.shape[1]


  #測試用，記得條回 100和20
  my_callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=100,monitor = 'loss',mode='min',min_delta=0.0001,restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
  ]
  # callback
  my_callbacks2 = [
      tf.keras.callbacks.EarlyStopping(patience=50,monitor = 'accuracy',mode='max',min_delta=0.0001,restore_best_weights=True)

  ]


  def sampling(args):
      z_mean, z_log_var = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      # by default, random_normal has mean = 0 and std = 1.0
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + K.exp(0.5 * z_log_var) * epsilon


  latent_dim = input_dim*0.2
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)


  # AE+SVM+KNN5折交叉驗證

  sfolder = StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y)

  svm_acc = []; svm_auc = []; svm_time = []

  for k, (train, test) in enumerate(sfolder):

      #重新命名 Kfold的資料 .iloc
      kfold_x_train = x.iloc[train]
      kfold_y_train = y.iloc[train]
      kfold_x_test =  x.iloc[test]
      kfold_y_test =  y.iloc[test]

      #特徵選取時間開始
      start1 = time.process_time()

      #特徵選取
      input_layer = Input(shape = (input_dim, ))
      #e1
      encoded = BatchNormalization()(input_layer)
      encoded = Dense(input_dim*0.6, activation='relu')(encoded)

      #e3
      encoded = BatchNormalization()(encoded)
      z_mean = enc_mean(encoded)
      z_log_var = enc_log_var(encoded)
      z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

      #D2
      decoded = BatchNormalization()(z)
      decoded = Dense(input_dim*0.6, activation='relu')(decoded)

      #output
      decoded = BatchNormalization()(decoded)
      output_layer = Dense(input_dim, activation='sigmoid')(decoded)

      #model
      VAE = Model(input_layer, output_layer)
      #softmax
      Soft_output = Dense(nb_classes, activation='softmax')(z_mean)
      ae_soft = Model(input_layer, Soft_output)

      #loss
      reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)

      kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
      kl_loss = K.sum(kl_loss, axis=-1)
      kl_loss *= -0.5

      vae_loss = K.mean(reconstruction_loss + kl_loss)

      VAE.add_loss(vae_loss)

      #compile
      VAE.compile(optimizer='adam', loss='mse')
      VAE.fit(kfold_x_train, kfold_x_train,
                  epochs=1000,
                  batch_size=16,
                  shuffle=True,
                  callbacks=my_callbacks,
                  )


      #softmax
      ae_soft.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      ae_soft.fit(kfold_x_train, kfold_y_train,
              batch_size=16,
              epochs=1000,
              shuffle=True,
              callbacks=my_callbacks2,
              validation_data=(kfold_x_test, kfold_y_test))

      y_predict = np.argmax(ae_soft.predict(kfold_x_test), axis=-1)

      #標籤轉換
      y_predict_m=label_binarize(y_predict, classes=yy)
      kfold_y_test_m=label_binarize(kfold_y_test, classes=yy)

      #分數
      acc = accuracy_score(kfold_y_test_m, y_predict_m)
      auc = roc_auc_score(kfold_y_test_m, y_predict_m,multi_class="ovo")

      #特徵選取時間結束
      end1 = time.process_time()
      tt1=end1 - start1

      #存各項數值
      svm_acc.append(acc); svm_auc.append(auc); svm_time.append(tt1)

  a=np.mean(svm_acc)
  b=np.mean(svm_auc)
  c=np.mean(svm_time)


  print(a)
  print(b)
  print(c)
  print("--------end--------")

  new=pd.DataFrame({'type':'(vae)(2-4)',
        'dataset':ff,
        'svm_acc':a,
        'svm_auc':b,
        'svm_time':c,
        },
                 index=[1])

  ansc45knn=ansc45knn.append(new)
  ansc45knn.to_csv('VAE/(ans)vae(前測)(2-4).csv',index=False,encoding='utf-8')

#vae(前測)(3-1)
fliename=['(03)sonar','(08)arrhythmia','(14)oh15.wc']
ansc45knn = pd.DataFrame()
for ff in fliename:

  print(ff)
  file=ff
  df = pd.read_csv('org/'+file+'.csv')
  labelencoder = LabelEncoder()
  df['Class'] = labelencoder.fit_transform(df['Class'])

  y = df['Class']
  x = df.drop(['Class'], axis=1)
  x_c = df.drop(['Class'], axis=1)#取得columns名稱
  yy=np.unique(y)#多分類AUC轉換使用

  #資料正規化
  minmax = preprocessing.MinMaxScaler()
  x = minmax.fit_transform(x)
  x=pd.DataFrame(x,columns = x_c.columns)

  #算維度
  input_dim = x.shape[1]
  y_train_class = np_utils.to_categorical(y)
  nb_classes = y_train_class.shape[1]


  #測試用，記得條回 100和20
  my_callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=100,monitor = 'loss',mode='min',min_delta=0.0001,restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
  ]
  # callback
  my_callbacks2 = [
      tf.keras.callbacks.EarlyStopping(patience=50,monitor = 'accuracy',mode='max',min_delta=0.0001,restore_best_weights=True)

  ]


  def sampling(args):
      z_mean, z_log_var = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      # by default, random_normal has mean = 0 and std = 1.0
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + K.exp(0.5 * z_log_var) * epsilon


  latent_dim = input_dim*0.7
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)


  # AE+SVM+KNN5折交叉驗證

  sfolder = StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y)

  svm_acc = []; svm_auc = []; svm_time = []

  for k, (train, test) in enumerate(sfolder):

      #重新命名 Kfold的資料 .iloc
      kfold_x_train = x.iloc[train]
      kfold_y_train = y.iloc[train]
      kfold_x_test =  x.iloc[test]
      kfold_y_test =  y.iloc[test]

      #特徵選取時間開始
      start1 = time.process_time()

      #特徵選取
      input_layer = Input(shape = (input_dim, ))
      #e1
      encoded = BatchNormalization()(input_layer)
      encoded = Dense(input_dim*0.9, activation='relu')(encoded)
      #e2
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.8, activation='relu')(encoded)

      #e3
      encoded = BatchNormalization()(encoded)
      z_mean = enc_mean(encoded)
      z_log_var = enc_log_var(encoded)
      z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])


      #D2
      decoded = BatchNormalization()(z)
      decoded = Dense(input_dim*0.8, activation='relu')(decoded)
      #D2
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.9, activation='relu')(decoded)
      #output
      decoded = BatchNormalization()(decoded)
      output_layer = Dense(input_dim, activation='sigmoid')(decoded)

      #model
      VAE = Model(input_layer, output_layer)
      #softmax
      Soft_output = Dense(nb_classes, activation='softmax')(z_mean)
      ae_soft = Model(input_layer, Soft_output)

      #loss
      reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)

      kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
      kl_loss = K.sum(kl_loss, axis=-1)
      kl_loss *= -0.5

      vae_loss = K.mean(reconstruction_loss + kl_loss)

      VAE.add_loss(vae_loss)

      #compile
      VAE.compile(optimizer='adam', loss='mse')
      VAE.fit(kfold_x_train, kfold_x_train,
                  epochs=1000,
                  batch_size=16,
                  shuffle=True,
                  callbacks=my_callbacks,
                  )


      #softmax
      ae_soft.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      ae_soft.fit(kfold_x_train, kfold_y_train,
              batch_size=16,
              epochs=1000,
              shuffle=True,
              callbacks=my_callbacks2,
              validation_data=(kfold_x_test, kfold_y_test))

      y_predict = np.argmax(ae_soft.predict(kfold_x_test), axis=-1)

      #標籤轉換
      y_predict_m=label_binarize(y_predict, classes=yy)
      kfold_y_test_m=label_binarize(kfold_y_test, classes=yy)

      #分數
      acc = accuracy_score(kfold_y_test_m, y_predict_m)
      auc = roc_auc_score(kfold_y_test_m, y_predict_m,multi_class="ovo")

      #特徵選取時間結束
      end1 = time.process_time()
      tt1=end1 - start1

      #存各項數值
      svm_acc.append(acc); svm_auc.append(auc); svm_time.append(tt1)

  a=np.mean(svm_acc)
  b=np.mean(svm_auc)
  c=np.mean(svm_time)


  print(a)
  print(b)
  print(c)
  print("--------end--------")

  new=pd.DataFrame({'type':'(vae)(3-1)',
        'dataset':ff,
        'svm_acc':a,
        'svm_auc':b,
        'svm_time':c,
        },
                 index=[1])

  ansc45knn=ansc45knn.append(new)
  ansc45knn.to_csv('VAE/(ans)vae(前測)(3-1).csv',index=False,encoding='utf-8')

#vae(前測)(3-2)
fliename=['(03)sonar','(08)arrhythmia','(14)oh15.wc']
ansc45knn = pd.DataFrame()
for ff in fliename:

  print(ff)
  file=ff
  df = pd.read_csv('org/'+file+'.csv')
  labelencoder = LabelEncoder()
  df['Class'] = labelencoder.fit_transform(df['Class'])

  y = df['Class']
  x = df.drop(['Class'], axis=1)
  x_c = df.drop(['Class'], axis=1)#取得columns名稱
  yy=np.unique(y)#多分類AUC轉換使用

  #資料正規化
  minmax = preprocessing.MinMaxScaler()
  x = minmax.fit_transform(x)
  x=pd.DataFrame(x,columns = x_c.columns)

  #算維度
  input_dim = x.shape[1]
  y_train_class = np_utils.to_categorical(y)
  nb_classes = y_train_class.shape[1]


  #測試用，記得條回 100和20
  my_callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=100,monitor = 'loss',mode='min',min_delta=0.0001,restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
  ]
  # callback
  my_callbacks2 = [
      tf.keras.callbacks.EarlyStopping(patience=50,monitor = 'accuracy',mode='max',min_delta=0.0001,restore_best_weights=True)

  ]

  def sampling(args):
      z_mean, z_log_var = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      # by default, random_normal has mean = 0 and std = 1.0
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + K.exp(0.5 * z_log_var) * epsilon


  latent_dim = input_dim*0.4
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)


  # AE+SVM+KNN5折交叉驗證

  sfolder = StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y)

  svm_acc = []; svm_auc = []; svm_time = []

  for k, (train, test) in enumerate(sfolder):

      #重新命名 Kfold的資料 .iloc
      kfold_x_train = x.iloc[train]
      kfold_y_train = y.iloc[train]
      kfold_x_test =  x.iloc[test]
      kfold_y_test =  y.iloc[test]

      #特徵選取時間開始
      start1 = time.process_time()

      #特徵選取
      input_layer = Input(shape = (input_dim, ))
      #e1
      encoded = BatchNormalization()(input_layer)
      encoded = Dense(input_dim*0.8, activation='relu')(encoded)
      #e2
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.6, activation='relu')(encoded)
      #e3
      encoded = BatchNormalization()(encoded)
      z_mean = enc_mean(encoded)
      z_log_var = enc_log_var(encoded)
      z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

      #D2
      decoded = BatchNormalization()(z)
      decoded = Dense(input_dim*0.6, activation='relu')(decoded)
      #D2
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.8, activation='relu')(decoded)
      #output
      decoded = BatchNormalization()(decoded)
      output_layer = Dense(input_dim, activation='sigmoid')(decoded)

      #model
      VAE = Model(input_layer, output_layer)
      #softmax
      Soft_output = Dense(nb_classes, activation='softmax')(z_mean)
      ae_soft = Model(input_layer, Soft_output)

      #loss
      reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)

      kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
      kl_loss = K.sum(kl_loss, axis=-1)
      kl_loss *= -0.5

      vae_loss = K.mean(reconstruction_loss + kl_loss)

      VAE.add_loss(vae_loss)

      #compile
      VAE.compile(optimizer='adam', loss='mse')
      VAE.fit(kfold_x_train, kfold_x_train,
                  epochs=1000,
                  batch_size=16,
                  shuffle=True,
                  callbacks=my_callbacks,
                  )


      #softmax
      ae_soft.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      ae_soft.fit(kfold_x_train, kfold_y_train,
              batch_size=16,
              epochs=1000,
              shuffle=True,
              callbacks=my_callbacks2,
              validation_data=(kfold_x_test, kfold_y_test))

      y_predict = np.argmax(ae_soft.predict(kfold_x_test), axis=-1)

      #標籤轉換
      y_predict_m=label_binarize(y_predict, classes=yy)
      kfold_y_test_m=label_binarize(kfold_y_test, classes=yy)

      #分數
      acc = accuracy_score(kfold_y_test_m, y_predict_m)
      auc = roc_auc_score(kfold_y_test_m, y_predict_m,multi_class="ovo")

      #特徵選取時間結束
      end1 = time.process_time()
      tt1=end1 - start1

      #存各項數值
      svm_acc.append(acc); svm_auc.append(auc); svm_time.append(tt1)

  a=np.mean(svm_acc)
  b=np.mean(svm_auc)
  c=np.mean(svm_time)


  print(a)
  print(b)
  print(c)
  print("--------end--------")

  new=pd.DataFrame({'type':'(vae)(3-2)',
        'dataset':ff,
        'svm_acc':a,
        'svm_auc':b,
        'svm_time':c,
        },
                 index=[1])

  ansc45knn=ansc45knn.append(new)
  ansc45knn.to_csv('VAE/(ans)vae(前測)(3-2).csv',index=False,encoding='utf-8')

#vae(前測)(3-3)
fliename=['(03)sonar','(08)arrhythmia','(14)oh15.wc']
ansc45knn = pd.DataFrame()
for ff in fliename:

  print(ff)
  file=ff
  df = pd.read_csv('org/'+file+'.csv')
  labelencoder = LabelEncoder()
  df['Class'] = labelencoder.fit_transform(df['Class'])

  y = df['Class']
  x = df.drop(['Class'], axis=1)
  x_c = df.drop(['Class'], axis=1)#取得columns名稱
  yy=np.unique(y)#多分類AUC轉換使用

  #資料正規化
  minmax = preprocessing.MinMaxScaler()
  x = minmax.fit_transform(x)
  x=pd.DataFrame(x,columns = x_c.columns)

  #算維度
  input_dim = x.shape[1]
  y_train_class = np_utils.to_categorical(y)
  nb_classes = y_train_class.shape[1]


  #測試用，記得條回 100和20
  my_callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=100,monitor = 'loss',mode='min',min_delta=0.0001,restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
  ]
  # callback
  my_callbacks2 = [
      tf.keras.callbacks.EarlyStopping(patience=50,monitor = 'accuracy',mode='max',min_delta=0.0001,restore_best_weights=True)

  ]

  def sampling(args):
      z_mean, z_log_var = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      # by default, random_normal has mean = 0 and std = 1.0
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + K.exp(0.5 * z_log_var) * epsilon


  latent_dim = input_dim*0.1
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)


  # AE+SVM+KNN5折交叉驗證

  sfolder = StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y)

  svm_acc = []; svm_auc = []; svm_time = []

  for k, (train, test) in enumerate(sfolder):

      #重新命名 Kfold的資料 .iloc
      kfold_x_train = x.iloc[train]
      kfold_y_train = y.iloc[train]
      kfold_x_test =  x.iloc[test]
      kfold_y_test =  y.iloc[test]

      #特徵選取時間開始
      start1 = time.process_time()

      #特徵選取
      input_layer = Input(shape = (input_dim, ))
      #e1
      encoded = BatchNormalization()(input_layer)
      encoded = Dense(input_dim*0.7, activation='relu')(encoded)
      #e2
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.4, activation='relu')(encoded)
      #e3
      encoded = BatchNormalization()(encoded)
      z_mean = enc_mean(encoded)
      z_log_var = enc_log_var(encoded)
      z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

      #D2
      decoded = BatchNormalization()(z)
      decoded = Dense(input_dim*0.4, activation='relu')(decoded)
      #D2
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.7, activation='relu')(decoded)
      #output
      decoded = BatchNormalization()(decoded)
      output_layer = Dense(input_dim, activation='sigmoid')(decoded)

      #model
      VAE = Model(input_layer, output_layer)
      #softmax
      Soft_output = Dense(nb_classes, activation='softmax')(z_mean)
      ae_soft = Model(input_layer, Soft_output)

      #loss
      reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)

      kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
      kl_loss = K.sum(kl_loss, axis=-1)
      kl_loss *= -0.5

      vae_loss = K.mean(reconstruction_loss + kl_loss)

      VAE.add_loss(vae_loss)

      #compile
      VAE.compile(optimizer='adam', loss='mse')
      VAE.fit(kfold_x_train, kfold_x_train,
                  epochs=1000,
                  batch_size=16,
                  shuffle=True,
                  callbacks=my_callbacks,
                  )


      #softmax
      ae_soft.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      ae_soft.fit(kfold_x_train, kfold_y_train,
              batch_size=16,
              epochs=1000,
              shuffle=True,
              callbacks=my_callbacks2,
              validation_data=(kfold_x_test, kfold_y_test))

      y_predict = np.argmax(ae_soft.predict(kfold_x_test), axis=-1)

      #標籤轉換
      y_predict_m=label_binarize(y_predict, classes=yy)
      kfold_y_test_m=label_binarize(kfold_y_test, classes=yy)

      #分數
      acc = accuracy_score(kfold_y_test_m, y_predict_m)
      auc = roc_auc_score(kfold_y_test_m, y_predict_m,multi_class="ovo")

      #特徵選取時間結束
      end1 = time.process_time()
      tt1=end1 - start1

      #存各項數值
      svm_acc.append(acc); svm_auc.append(auc); svm_time.append(tt1)

  a=np.mean(svm_acc)
  b=np.mean(svm_auc)
  c=np.mean(svm_time)


  print(a)
  print(b)
  print(c)
  print("--------end--------")

  new=pd.DataFrame({'type':'(vae)(3-3)',
        'dataset':ff,
        'svm_acc':a,
        'svm_auc':b,
        'svm_time':c,
        },
                 index=[1])

  ansc45knn=ansc45knn.append(new)
  ansc45knn.to_csv('VAE/(ans)vae(前測)(3-3).csv',index=False,encoding='utf-8')

#vae(前測)(4-1)
fliename=['(03)sonar','(08)arrhythmia','(14)oh15.wc']
ansc45knn = pd.DataFrame()
for ff in fliename:

  print(ff)
  file=ff
  df = pd.read_csv('org/'+file+'.csv')
  labelencoder = LabelEncoder()
  df['Class'] = labelencoder.fit_transform(df['Class'])

  y = df['Class']
  x = df.drop(['Class'], axis=1)
  x_c = df.drop(['Class'], axis=1)#取得columns名稱
  yy=np.unique(y)#多分類AUC轉換使用

  #資料正規化
  minmax = preprocessing.MinMaxScaler()
  x = minmax.fit_transform(x)
  x=pd.DataFrame(x,columns = x_c.columns)

  #算維度
  input_dim = x.shape[1]
  y_train_class = np_utils.to_categorical(y)
  nb_classes = y_train_class.shape[1]


  #測試用，記得條回 100和20
  my_callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=100,monitor = 'loss',mode='min',min_delta=0.0001,restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
  ]
  # callback
  my_callbacks2 = [
      tf.keras.callbacks.EarlyStopping(patience=50,monitor = 'accuracy',mode='max',min_delta=0.0001,restore_best_weights=True)

  ]


  def sampling(args):
      z_mean, z_log_var = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      # by default, random_normal has mean = 0 and std = 1.0
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + K.exp(0.5 * z_log_var) * epsilon


  latent_dim = input_dim*0.6
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)


  # AE+SVM+KNN5折交叉驗證

  sfolder = StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y)

  svm_acc = []; svm_auc = []; svm_time = []

  for k, (train, test) in enumerate(sfolder):

      #重新命名 Kfold的資料 .iloc
      kfold_x_train = x.iloc[train]
      kfold_y_train = y.iloc[train]
      kfold_x_test =  x.iloc[test]
      kfold_y_test =  y.iloc[test]

      #特徵選取時間開始
      start1 = time.process_time()

      #特徵選取
      input_layer = Input(shape = (input_dim, ))
      #e1
      encoded = BatchNormalization()(input_layer)
      encoded = Dense(input_dim*0.9, activation='relu')(encoded)
      #e2
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.8, activation='relu')(encoded)
      #e3
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.7, activation='relu')(encoded)
      #e4
      encoded = BatchNormalization()(encoded)
      z_mean = enc_mean(encoded)
      z_log_var = enc_log_var(encoded)
      z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

      #D3
      decoded = BatchNormalization()(z)
      decoded = Dense(input_dim*0.7, activation='relu')(decoded)
      #D2
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.8, activation='relu')(decoded)
      #D1
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.9, activation='relu')(decoded)
      #output
      decoded = BatchNormalization()(decoded)
      output_layer = Dense(input_dim, activation='sigmoid')(decoded)

      #model
      VAE = Model(input_layer, output_layer)
      #softmax
      Soft_output = Dense(nb_classes, activation='softmax')(z_mean)
      ae_soft = Model(input_layer, Soft_output)

      #loss
      reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)

      kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
      kl_loss = K.sum(kl_loss, axis=-1)
      kl_loss *= -0.5

      vae_loss = K.mean(reconstruction_loss + kl_loss)

      VAE.add_loss(vae_loss)

      #compile
      VAE.compile(optimizer='adam', loss='mse')
      VAE.fit(kfold_x_train, kfold_x_train,
                  epochs=1000,
                  batch_size=16,
                  shuffle=True,
                  callbacks=my_callbacks,
                  )


      #softmax
      ae_soft.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      ae_soft.fit(kfold_x_train, kfold_y_train,
              batch_size=16,
              epochs=1000,
              shuffle=True,
              callbacks=my_callbacks2,
              validation_data=(kfold_x_test, kfold_y_test))

      y_predict = np.argmax(ae_soft.predict(kfold_x_test), axis=-1)

      #標籤轉換
      y_predict_m=label_binarize(y_predict, classes=yy)
      kfold_y_test_m=label_binarize(kfold_y_test, classes=yy)

      #分數
      acc = accuracy_score(kfold_y_test_m, y_predict_m)
      auc = roc_auc_score(kfold_y_test_m, y_predict_m,multi_class="ovo")

      #特徵選取時間結束
      end1 = time.process_time()
      tt1=end1 - start1

      #存各項數值
      svm_acc.append(acc); svm_auc.append(auc); svm_time.append(tt1)

  a=np.mean(svm_acc)
  b=np.mean(svm_auc)
  c=np.mean(svm_time)


  print(a)
  print(b)
  print(c)
  print("--------end--------")

  new=pd.DataFrame({'type':'(vae)(4-1)',
        'dataset':ff,
        'svm_acc':a,
        'svm_auc':b,
        'svm_time':c,
        },
                 index=[1])

  ansc45knn=ansc45knn.append(new)
  ansc45knn.to_csv('VAE/(ans)vae(前測)(4-1).csv',index=False,encoding='utf-8')

#vae(前測)(4-2)
fliename=['(03)sonar','(08)arrhythmia','(14)oh15.wc']
ansc45knn = pd.DataFrame()
for ff in fliename:

  print(ff)
  file=ff
  df = pd.read_csv('org/'+file+'.csv')
  labelencoder = LabelEncoder()
  df['Class'] = labelencoder.fit_transform(df['Class'])

  y = df['Class']
  x = df.drop(['Class'], axis=1)
  x_c = df.drop(['Class'], axis=1)#取得columns名稱
  yy=np.unique(y)#多分類AUC轉換使用

  #資料正規化
  minmax = preprocessing.MinMaxScaler()
  x = minmax.fit_transform(x)
  x=pd.DataFrame(x,columns = x_c.columns)

  #算維度
  input_dim = x.shape[1]
  y_train_class = np_utils.to_categorical(y)
  nb_classes = y_train_class.shape[1]


  #測試用，記得條回 100和20
  my_callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=100,monitor = 'loss',mode='min',min_delta=0.0001,restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
  ]
  # callback
  my_callbacks2 = [
      tf.keras.callbacks.EarlyStopping(patience=50,monitor = 'accuracy',mode='max',min_delta=0.0001,restore_best_weights=True)

  ]

  def sampling(args):
      z_mean, z_log_var = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      # by default, random_normal has mean = 0 and std = 1.0
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + K.exp(0.5 * z_log_var) * epsilon


  latent_dim = input_dim*0.2
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)


  # AE+SVM+KNN5折交叉驗證

  sfolder = StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y)

  svm_acc = []; svm_auc = []; svm_time = []

  for k, (train, test) in enumerate(sfolder):

      #重新命名 Kfold的資料 .iloc
      kfold_x_train = x.iloc[train]
      kfold_y_train = y.iloc[train]
      kfold_x_test =  x.iloc[test]
      kfold_y_test =  y.iloc[test]

      #特徵選取時間開始
      start1 = time.process_time()

      #特徵選取
      input_layer = Input(shape = (input_dim, ))
      #e1
      encoded = BatchNormalization()(input_layer)
      encoded = Dense(input_dim*0.8, activation='relu')(encoded)
      #e2
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.6, activation='relu')(encoded)
      #e3
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.4, activation='relu')(encoded)
      #e4
      encoded = BatchNormalization()(encoded)
      z_mean = enc_mean(encoded)
      z_log_var = enc_log_var(encoded)
      z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

      #D3
      decoded = BatchNormalization()(z)
      decoded = Dense(input_dim*0.4, activation='relu')(decoded)
      #D2
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.6, activation='relu')(decoded)
      #D1
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.8, activation='relu')(decoded)
      #output
      decoded = BatchNormalization()(decoded)
      output_layer = Dense(input_dim, activation='sigmoid')(decoded)

      #model
      VAE = Model(input_layer, output_layer)
      #softmax
      Soft_output = Dense(nb_classes, activation='softmax')(z_mean)
      ae_soft = Model(input_layer, Soft_output)

      #loss
      reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)

      kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
      kl_loss = K.sum(kl_loss, axis=-1)
      kl_loss *= -0.5

      vae_loss = K.mean(reconstruction_loss + kl_loss)

      VAE.add_loss(vae_loss)

      #compile
      VAE.compile(optimizer='adam', loss='mse')
      VAE.fit(kfold_x_train, kfold_x_train,
                  epochs=1000,
                  batch_size=16,
                  shuffle=True,
                  callbacks=my_callbacks,
                  )


      #softmax
      ae_soft.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      ae_soft.fit(kfold_x_train, kfold_y_train,
              batch_size=16,
              epochs=1000,
              shuffle=True,
              callbacks=my_callbacks2,
              validation_data=(kfold_x_test, kfold_y_test))

      y_predict = np.argmax(ae_soft.predict(kfold_x_test), axis=-1)

      #標籤轉換
      y_predict_m=label_binarize(y_predict, classes=yy)
      kfold_y_test_m=label_binarize(kfold_y_test, classes=yy)

      #分數
      acc = accuracy_score(kfold_y_test_m, y_predict_m)
      auc = roc_auc_score(kfold_y_test_m, y_predict_m,multi_class="ovo")

      #特徵選取時間結束
      end1 = time.process_time()
      tt1=end1 - start1

      #存各項數值
      svm_acc.append(acc); svm_auc.append(auc); svm_time.append(tt1)

  a=np.mean(svm_acc)
  b=np.mean(svm_auc)
  c=np.mean(svm_time)


  print(a)
  print(b)
  print(c)
  print("--------end--------")

  new=pd.DataFrame({'type':'(vae)(4-2)',
        'dataset':ff,
        'svm_acc':a,
        'svm_auc':b,
        'svm_time':c,
        },
                 index=[1])

  ansc45knn=ansc45knn.append(new)
  ansc45knn.to_csv('VAE/(ans)vae(前測)(4-2).csv',index=False,encoding='utf-8')

#vae(前測)(5-1)
fliename=['(03)sonar','(08)arrhythmia','(14)oh15.wc']
ansc45knn = pd.DataFrame()
for ff in fliename:

  print(ff)
  file=ff
  df = pd.read_csv('org/'+file+'.csv')
  labelencoder = LabelEncoder()
  df['Class'] = labelencoder.fit_transform(df['Class'])

  y = df['Class']
  x = df.drop(['Class'], axis=1)
  x_c = df.drop(['Class'], axis=1)#取得columns名稱
  yy=np.unique(y)#多分類AUC轉換使用

  #資料正規化
  minmax = preprocessing.MinMaxScaler()
  x = minmax.fit_transform(x)
  x=pd.DataFrame(x,columns = x_c.columns)

  #算維度
  input_dim = x.shape[1]
  y_train_class = np_utils.to_categorical(y)
  nb_classes = y_train_class.shape[1]


  #測試用，記得條回 100和20
  my_callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=100,monitor = 'loss',mode='min',min_delta=0.0001,restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
  ]
  # callback
  my_callbacks2 = [
      tf.keras.callbacks.EarlyStopping(patience=50,monitor = 'accuracy',mode='max',min_delta=0.0001,restore_best_weights=True)

  ]


  def sampling(args):
      z_mean, z_log_var = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      # by default, random_normal has mean = 0 and std = 1.0
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + K.exp(0.5 * z_log_var) * epsilon


  latent_dim = input_dim*0.5
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)


  # AE+SVM+KNN5折交叉驗證

  sfolder = StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y)

  svm_acc = []; svm_auc = []; svm_time = []

  for k, (train, test) in enumerate(sfolder):

      #重新命名 Kfold的資料 .iloc
      kfold_x_train = x.iloc[train]
      kfold_y_train = y.iloc[train]
      kfold_x_test =  x.iloc[test]
      kfold_y_test =  y.iloc[test]

      #特徵選取時間開始
      start1 = time.process_time()

      #特徵選取
      input_layer = Input(shape = (input_dim, ))
      #e1
      encoded = BatchNormalization()(input_layer)
      encoded = Dense(input_dim*0.9, activation='relu')(encoded)
      #e2
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.8, activation='relu')(encoded)
      #e3
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.7, activation='relu')(encoded)
      #e4
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.6, activation='relu')(encoded)
      #e5
      encoded = BatchNormalization()(encoded)
      z_mean = enc_mean(encoded)
      z_log_var = enc_log_var(encoded)
      z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

      #D4
      decoded = BatchNormalization()(z)
      decoded = Dense(input_dim*0.6, activation='relu')(decoded)
      #D3
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.7, activation='relu')(decoded)
      #D2
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.8, activation='relu')(decoded)
      #D1
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.9, activation='relu')(decoded)
      #output
      decoded = BatchNormalization()(decoded)
      output_layer = Dense(input_dim, activation='sigmoid')(decoded)

      #model
      VAE = Model(input_layer, output_layer)
      #softmax
      Soft_output = Dense(nb_classes, activation='softmax')(z_mean)
      ae_soft = Model(input_layer, Soft_output)

      #loss
      reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)

      kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
      kl_loss = K.sum(kl_loss, axis=-1)
      kl_loss *= -0.5

      vae_loss = K.mean(reconstruction_loss + kl_loss)

      VAE.add_loss(vae_loss)

      #compile
      VAE.compile(optimizer='adam', loss='mse')
      VAE.fit(kfold_x_train, kfold_x_train,
                  epochs=1000,
                  batch_size=16,
                  shuffle=True,
                  callbacks=my_callbacks,
                  )


      #softmax
      ae_soft.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      ae_soft.fit(kfold_x_train, kfold_y_train,
              batch_size=16,
              epochs=1000,
              shuffle=True,
              callbacks=my_callbacks2,
              validation_data=(kfold_x_test, kfold_y_test))

      y_predict = np.argmax(ae_soft.predict(kfold_x_test), axis=-1)

      #標籤轉換
      y_predict_m=label_binarize(y_predict, classes=yy)
      kfold_y_test_m=label_binarize(kfold_y_test, classes=yy)

      #分數
      acc = accuracy_score(kfold_y_test_m, y_predict_m)
      auc = roc_auc_score(kfold_y_test_m, y_predict_m,multi_class="ovo")

      #特徵選取時間結束
      end1 = time.process_time()
      tt1=end1 - start1

      #存各項數值
      svm_acc.append(acc); svm_auc.append(auc); svm_time.append(tt1)

  a=np.mean(svm_acc)
  b=np.mean(svm_auc)
  c=np.mean(svm_time)


  print(a)
  print(b)
  print(c)
  print("--------end--------")

  new=pd.DataFrame({'type':'(vae)(5-1)',
        'dataset':ff,
        'svm_acc':a,
        'svm_auc':b,
        'svm_time':c,
        },
                 index=[1])

  ansc45knn=ansc45knn.append(new)
  ansc45knn.to_csv('VAE/(ans)vae(前測)(5-1).csv',index=False,encoding='utf-8')

#vae(前測)(6-1)
fliename=['(03)sonar','(08)arrhythmia','(14)oh15.wc']
ansc45knn = pd.DataFrame()
for ff in fliename:

  print(ff)
  file=ff
  df = pd.read_csv('org/'+file+'.csv')
  labelencoder = LabelEncoder()
  df['Class'] = labelencoder.fit_transform(df['Class'])

  y = df['Class']
  x = df.drop(['Class'], axis=1)
  x_c = df.drop(['Class'], axis=1)#取得columns名稱
  yy=np.unique(y)#多分類AUC轉換使用

  #資料正規化
  minmax = preprocessing.MinMaxScaler()
  x = minmax.fit_transform(x)
  x=pd.DataFrame(x,columns = x_c.columns)

  #算維度
  input_dim = x.shape[1]
  y_train_class = np_utils.to_categorical(y)
  nb_classes = y_train_class.shape[1]


  #測試用，記得條回 100和20
  my_callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=100,monitor = 'loss',mode='min',min_delta=0.0001,restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
  ]
  # callback
  my_callbacks2 = [
      tf.keras.callbacks.EarlyStopping(patience=50,monitor = 'accuracy',mode='max',min_delta=0.0001,restore_best_weights=True)

  ]


  def sampling(args):
      z_mean, z_log_var = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      # by default, random_normal has mean = 0 and std = 1.0
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + K.exp(0.5 * z_log_var) * epsilon


  latent_dim = input_dim*0.4
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)


  # AE+SVM+KNN5折交叉驗證

  sfolder = StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y)

  svm_acc = []; svm_auc = []; svm_time = []

  for k, (train, test) in enumerate(sfolder):

      #重新命名 Kfold的資料 .iloc
      kfold_x_train = x.iloc[train]
      kfold_y_train = y.iloc[train]
      kfold_x_test =  x.iloc[test]
      kfold_y_test =  y.iloc[test]

      #特徵選取時間開始
      start1 = time.process_time()

      #特徵選取
      input_layer = Input(shape = (input_dim, ))
      #e1
      encoded = BatchNormalization()(input_layer)
      encoded = Dense(input_dim*0.9, activation='relu')(encoded)
      #e2
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.8, activation='relu')(encoded)
      #e3
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.7, activation='relu')(encoded)
      #e4
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.6, activation='relu')(encoded)
      #e5
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.5, activation='relu')(encoded)
      #e6
      encoded = BatchNormalization()(encoded)
      z_mean = enc_mean(encoded)
      z_log_var = enc_log_var(encoded)
      z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

      #D5
      decoded = BatchNormalization()(z)
      decoded = Dense(input_dim*0.5, activation='relu')(decoded)
      #D4
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.6, activation='relu')(decoded)
      #D3
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.7, activation='relu')(decoded)
      #D2
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.8, activation='relu')(decoded)
      #D1
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.9, activation='relu')(decoded)
      #output
      decoded = BatchNormalization()(decoded)
      output_layer = Dense(input_dim, activation='sigmoid')(decoded)

      #model
      VAE = Model(input_layer, output_layer)
      #softmax
      Soft_output = Dense(nb_classes, activation='softmax')(z_mean)
      ae_soft = Model(input_layer, Soft_output)

      #loss
      reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)

      kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
      kl_loss = K.sum(kl_loss, axis=-1)
      kl_loss *= -0.5

      vae_loss = K.mean(reconstruction_loss + kl_loss)

      VAE.add_loss(vae_loss)

      #compile
      VAE.compile(optimizer='adam', loss='mse')
      VAE.fit(kfold_x_train, kfold_x_train,
                  epochs=1000,
                  batch_size=16,
                  shuffle=True,
                  callbacks=my_callbacks,
                  )


      #softmax
      ae_soft.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      ae_soft.fit(kfold_x_train, kfold_y_train,
              batch_size=16,
              epochs=1000,
              shuffle=True,
              callbacks=my_callbacks2,
              validation_data=(kfold_x_test, kfold_y_test))

      y_predict = np.argmax(ae_soft.predict(kfold_x_test), axis=-1)

      #標籤轉換
      y_predict_m=label_binarize(y_predict, classes=yy)
      kfold_y_test_m=label_binarize(kfold_y_test, classes=yy)

      #分數
      acc = accuracy_score(kfold_y_test_m, y_predict_m)
      auc = roc_auc_score(kfold_y_test_m, y_predict_m,multi_class="ovo")

      #特徵選取時間結束
      end1 = time.process_time()
      tt1=end1 - start1

      #存各項數值
      svm_acc.append(acc); svm_auc.append(auc); svm_time.append(tt1)

  a=np.mean(svm_acc)
  b=np.mean(svm_auc)
  c=np.mean(svm_time)


  print(a)
  print(b)
  print(c)
  print("--------end--------")

  new=pd.DataFrame({'type':'(vae)(6-1)',
        'dataset':ff,
        'svm_acc':a,
        'svm_auc':b,
        'svm_time':c,
        },
                 index=[1])

  ansc45knn=ansc45knn.append(new)
  ansc45knn.to_csv('VAE/(ans)vae(前測)(6-1).csv',index=False,encoding='utf-8')

#vae(前測)(7-1)
fliename=['(03)sonar','(08)arrhythmia','(14)oh15.wc']
ansc45knn = pd.DataFrame()
for ff in fliename:

  print(ff)
  file=ff
  df = pd.read_csv('org/'+file+'.csv')
  labelencoder = LabelEncoder()
  df['Class'] = labelencoder.fit_transform(df['Class'])

  y = df['Class']
  x = df.drop(['Class'], axis=1)
  x_c = df.drop(['Class'], axis=1)#取得columns名稱
  yy=np.unique(y)#多分類AUC轉換使用

  #資料正規化
  minmax = preprocessing.MinMaxScaler()
  x = minmax.fit_transform(x)
  x=pd.DataFrame(x,columns = x_c.columns)

  #算維度
  input_dim = x.shape[1]
  y_train_class = np_utils.to_categorical(y)
  nb_classes = y_train_class.shape[1]


  #測試用，記得條回 100和20
  my_callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=100,monitor = 'loss',mode='min',min_delta=0.0001,restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
  ]
  # callback
  my_callbacks2 = [
      tf.keras.callbacks.EarlyStopping(patience=50,monitor = 'accuracy',mode='max',min_delta=0.0001,restore_best_weights=True)

  ]


  def sampling(args):
      z_mean, z_log_var = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      # by default, random_normal has mean = 0 and std = 1.0
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + K.exp(0.5 * z_log_var) * epsilon


  latent_dim = input_dim*0.3
  enc_mean = Dense(latent_dim)
  enc_log_var = Dense(latent_dim)


  # AE+SVM+KNN5折交叉驗證

  sfolder = StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y)

  svm_acc = []; svm_auc = []; svm_time = []

  for k, (train, test) in enumerate(sfolder):

      #重新命名 Kfold的資料 .iloc
      kfold_x_train = x.iloc[train]
      kfold_y_train = y.iloc[train]
      kfold_x_test =  x.iloc[test]
      kfold_y_test =  y.iloc[test]

      #特徵選取時間開始
      start1 = time.process_time()

      #特徵選取
      input_layer = Input(shape = (input_dim, ))
      #e1
      encoded = BatchNormalization()(input_layer)
      encoded = Dense(input_dim*0.9, activation='relu')(encoded)
      #e2
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.8, activation='relu')(encoded)
      #e3
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.7, activation='relu')(encoded)
      #e4
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.6, activation='relu')(encoded)
      #e5
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.5, activation='relu')(encoded)
      #e6
      encoded = BatchNormalization()(encoded)
      encoded = Dense(input_dim*0.4, activation='relu')(encoded)
      #e7
      encoded = BatchNormalization()(encoded)
      z_mean = enc_mean(encoded)
      z_log_var = enc_log_var(encoded)
      z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

      #D6
      decoded = BatchNormalization()(z)
      decoded = Dense(input_dim*0.4, activation='relu')(decoded)
      #D5
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.5, activation='relu')(decoded)
      #D4
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.6, activation='relu')(decoded)
      #D3
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.7, activation='relu')(decoded)
      #D2
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.8, activation='relu')(decoded)
      #D1
      decoded = BatchNormalization()(decoded)
      decoded = Dense(input_dim*0.9, activation='relu')(decoded)
      #output
      decoded = BatchNormalization()(decoded)
      output_layer = Dense(input_dim, activation='sigmoid')(decoded)

      #model
      VAE = Model(input_layer, output_layer)
      #softmax
      Soft_output = Dense(nb_classes, activation='softmax')(z_mean)
      ae_soft = Model(input_layer, Soft_output)

      #loss
      reconstruction_loss = input_dim * losses.mean_squared_error(input_layer, output_layer)

      kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
      kl_loss = K.sum(kl_loss, axis=-1)
      kl_loss *= -0.5

      vae_loss = K.mean(reconstruction_loss + kl_loss)

      VAE.add_loss(vae_loss)

      #compile
      VAE.compile(optimizer='adam', loss='mse')
      VAE.fit(kfold_x_train, kfold_x_train,
                  epochs=1000,
                  batch_size=16,
                  shuffle=True,
                  callbacks=my_callbacks,
                  )


      #softmax
      ae_soft.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      ae_soft.fit(kfold_x_train, kfold_y_train,
              batch_size=16,
              epochs=1000,
              shuffle=True,
              callbacks=my_callbacks2,
              validation_data=(kfold_x_test, kfold_y_test))

      y_predict = np.argmax(ae_soft.predict(kfold_x_test), axis=-1)

      #標籤轉換
      y_predict_m=label_binarize(y_predict, classes=yy)
      kfold_y_test_m=label_binarize(kfold_y_test, classes=yy)

      #分數
      acc = accuracy_score(kfold_y_test_m, y_predict_m)
      auc = roc_auc_score(kfold_y_test_m, y_predict_m,multi_class="ovo")

      #特徵選取時間結束
      end1 = time.process_time()
      tt1=end1 - start1

      #存各項數值
      svm_acc.append(acc); svm_auc.append(auc); svm_time.append(tt1)

  a=np.mean(svm_acc)
  b=np.mean(svm_auc)
  c=np.mean(svm_time)


  print(a)
  print(b)
  print(c)
  print("--------end--------")

  new=pd.DataFrame({'type':'(vae)(7-1)',
        'dataset':ff,
        'svm_acc':a,
        'svm_auc':b,
        'svm_time':c,
        },
                 index=[1])

  ansc45knn=ansc45knn.append(new)
  ansc45knn.to_csv('VAE/(ans)vae(前測)(7-1).csv',index=False,encoding='utf-8')