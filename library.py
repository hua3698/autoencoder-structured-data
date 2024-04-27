from common import *

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

  # Label encode
  labelencoder = LabelEncoder()
  y_train = labelencoder.fit_transform(df_train['Class'])
  y_test = labelencoder.fit_transform(df_test['Class'])

  x_train = df_train.drop(['Class'], axis=1)
  x_test = df_test.drop(['Class'], axis=1)

  string_columns = x_train.select_dtypes(include=['object']).columns

  for col in string_columns:

      labelencoder = LabelEncoder()
      x_train[col] = labelencoder.fit_transform(df_train[col])
      x_test[col] = labelencoder.fit_transform(x_test[col])

  # 特徵縮放
  minmax = preprocessing.MinMaxScaler()
  x_train_minmax = minmax.fit_transform(x_train)
  x_test_minmax = minmax.fit_transform(x_test)

  x_train = pd.DataFrame(x_train_minmax, columns = x_train.columns)
  x_test = pd.DataFrame(x_test_minmax, columns = x_test.columns)

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

# autoencoder
ae_version = 'h2-20'

def train_ae(x_train, x_test):

  input_dim = x_train.shape[1]
  input_layer = Input(shape = (input_dim, ))
  print(input_layer)

  encoded = BatchNormalization()(input_layer)
  encoded = Dense(int(input_dim*0.6), activation='relu')(encoded)

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

# SAE
from keras import regularizers

def train_sae(x_train, x_test):

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