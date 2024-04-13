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
#svm
from sklearn.svm import SVC
#KNN
from sklearn.neighbors import KNeighborsClassifier
#AE
from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Dropout, Input,BatchNormalization

#MLP juypter
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input,BatchNormalization

#AE+SVM+KNN
fliename=['(01)SPECTF','(02A)segmentationData','(03)sonar','(04A)meu-mobile-ksd-2016','(05A)urban_land_cover','(06)musk_clean1','(07)SCADI','(08)arrhythmia','(09A)madelon','(10)secom','(11A)gastroentrology','(12)Yale','(13)colon','(14)oh15.wc','(15)oh10.wc','(16)leukemia','(17A)Amazon_initial_50_30_10000','(18)CLL_SUB_111','(19)SMK_CAN_187','(20)GLI_85']
ansc45knn = pd.DataFrame()
for ff in fliename:

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

  #計算維度
  input_dim = x.shape[1]
  y_train_class = np_utils.to_categorical(y)
  nb_classes = y_train_class.shape[1]

  #callback
  my_callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=100,monitor='loss',mode='min',min_delta=0.0001,restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
  ]


  # AE+SVM+KNN5折交叉驗證
  sfolder = StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x,y)

  f_time = []; f_dim = []
  svm_acc = []; svm_auc = []; svm_time = []
  knn_acc = []; knn_auc = []; knn_time = []

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
      encoded_end = Dense(input_dim*0.2, activation='relu')(encoded)
      #D3
      decoded = BatchNormalization()(encoded_end)
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
      ae = Model(input_layer, output_layer)
      ae_e = Model(input_layer, encoded_end)

      ae.compile(optimizer='adam', loss='mse')
      ae.fit(kfold_x_train, kfold_x_train,
                  epochs=1000,
                  batch_size=16,
                  shuffle=True,
                  callbacks=my_callbacks)

      x_train_ae = ae_e.predict(kfold_x_train)
      x_test_ae = ae_e.predict(kfold_x_test)
      dim_fs = x_test_ae.shape[1]


      #特徵選取時間結束
      end1 = time.process_time()
      tt1 = end1 - start1


      #存特徵選取後的檔案
      x_train_ae_df = pd.DataFrame(x_train_ae)
      x_test_ae_df = pd.DataFrame(x_test_ae)
      y_ans_train = kfold_y_train.to_numpy()
      y_ans_train = pd.DataFrame(y_ans_train,columns = ["Class"])
      y_ans_test = kfold_y_test.to_numpy()
      y_ans_test = pd.DataFrame(y_ans_test,columns = ["Class"])
      df1 = x_train_ae_df.merge(y_ans_train, how='left', left_index=True, right_index=True)
      df2 = x_test_ae_df.merge(y_ans_test, how='left', left_index=True, right_index=True)
      df1.to_csv('AE+SVM+KNN/' + str(file) + '_ae_train_' + str(k+1) + '.csv', index=False, encoding='utf-8')
      df2.to_csv('AE+SVM+KNN/' + str(file) + '_ae_test_' + str(k+1) + '.csv', index=False, encoding='utf-8')

      #存各項數值
      f_dim.append(dim_fs); f_time.append(tt1)

      #分類器時間開始
      start2 = time.process_time()

      #分類器
      svm = SVC(kernel='linear')
      svm.fit(x_train_ae, kfold_y_train)
      y_predict = svm.predict(x_test_ae)

      #標籤轉換
      y_predict_m=label_binarize(y_predict, classes=yy)
      kfold_y_test_m=label_binarize(kfold_y_test, classes=yy)

      #分數
      acc = accuracy_score(kfold_y_test_m, y_predict_m)
      auc = roc_auc_score(kfold_y_test_m, y_predict_m,multi_class="ovo")

      #分類器時間結束
      end2 = time.process_time()
      tt2=end2 - start2

      #存ANS檔案
      y_guess = pd.DataFrame(y_predict, columns = ["guess"])
      y_ans = kfold_y_test.to_numpy()
      y_ans = pd.DataFrame(y_ans, columns = ["true"])
      test_ans = y_ans.merge(y_guess, how='left', left_index=True, right_index=True)
      test_ans.to_csv('AE+SVM+KNN/'+str(file)+'_svm+ae_ans_'+str(k+1)+'.csv',index=False,encoding='utf-8')

      #存各項數值
      svm_acc.append(acc); svm_auc.append(auc); svm_time.append(tt2)

      #分類器時間開始
      start3 = time.process_time()

      #分類器
      knn = KNeighborsClassifier()
      knn.fit(x_train_ae, kfold_y_train)
      y_predict = knn.predict(x_test_ae)

      #標籤轉換
      y_predict_m = label_binarize(y_predict, classes=yy)
      kfold_y_test_m = label_binarize(kfold_y_test, classes=yy)

      #分數
      acc = accuracy_score(kfold_y_test_m, y_predict_m)
      auc = roc_auc_score(kfold_y_test_m, y_predict_m,multi_class="ovo")

      #時間結束
      end3 = time.process_time()
      tt3 = end3 - start3

      #存檔案
      y_guess = pd.DataFrame(y_predict, columns = ["guess"])
      y_ans = kfold_y_test.to_numpy()
      y_ans = pd.DataFrame(y_ans, columns = ["true"])
      test_ans = y_ans.merge(y_guess, how='left', left_index=True, right_index=True)
      test_ans.to_csv('AE+SVM+KNN/' + str(file) + '_knn+ae_ans_' + str(k+1) + '.csv', index=False, encoding='utf-8')

      #存各項數值
      knn_acc.append(acc); knn_auc.append(auc); knn_time.append(tt3)



  #計算平均分數
  a=np.mean(svm_acc)
  b=np.mean(svm_auc)
  c=np.mean(svm_time)
  d=np.mean(knn_acc)
  e=np.mean(knn_auc)
  f=np.mean(knn_time)

  g=np.mean(f_dim)
  h=g/input_dim
  i=np.mean(f_time)

  #存結果檔案
  new=pd.DataFrame({'dataset':ff,
        'svm_acc': a,
        'svm_auc': b,
        'svm_time': c,
        'knn_acc': d,
        'knn_auc': e,
        'knn_time': f,
        'ae_dim': g,
        'ae_dimp': h,
        'ae_time': i,
        },
        index=[1])

  ansc45knn=ansc45knn.append(new)
  ansc45knn.to_csv('AE+SVM+KNN/(ans)AE+SVM+KNN.csv',index=False,encoding='utf-8')