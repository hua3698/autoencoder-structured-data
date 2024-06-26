# VAE
import tensorflow.keras.backend as K
from tensorflow.keras import losses
from tensorflow.keras.layers import Dense, Activation, Input,BatchNormalization,Lambda

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


result = pd.DataFrame()

for idx, dataset in enumerate(datasets):

  labelencoder = LabelEncoder()
  minmax = preprocessing.MinMaxScaler()

  f_time = []; f_dim = []; total_time = 0
  svm_acc = []; svm_auc = []; svm_time = []; knn_acc = []; knn_auc = []; knn_time = []

  for times in range(1,6):

    training = "{}-5-{}{}.dat".format(dataset, times, 'tra')
    testing = "{}-5-{}{}.dat".format(dataset, times, 'tst')

    df_train = pd.read_csv('drive/My Drive/Colab Notebooks/datasets/' + dataset + '/' + training, delimiter=',')
    df_test = pd.read_csv('drive/My Drive/Colab Notebooks/datasets/' + dataset + '/' + testing, delimiter=',')

    y_train = labelencoder.fit_transform(df_train['Class'])
    y_test = labelencoder.fit_transform(df_test['Class'])

    x_train = df_train.drop(['Class'], axis=1)
    x_test = df_test.drop(['Class'], axis=1)

    #特徵縮放
    x_train_minmax = minmax.fit_transform(x_train)
    x_test_minmax = minmax.fit_transform(x_test)

    x_train = pd.DataFrame(x_train_minmax, columns = x_train.columns)
    x_test = pd.DataFrame(x_test_minmax, columns = x_test.columns)

    unique_y = transfer_y(y_train)

    #算維度
    input_dim = x_train.shape[1]
    y_train_class = tf.keras.utils.to_categorical(y_train)
    nb_classes = y_train_class.shape[1]

    #callback
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=100,monitor='loss',mode='min',min_delta=0.0001,restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,min_delta=0.0001,mode='min')
    ]

    latent_dim = input_dim
    enc_mean = Dense(latent_dim)
    enc_log_var = Dense(latent_dim)

    #特徵選取時間開始
    start1 = time.process_time()

    #特徵選取
    input_layer = Input(shape = (input_dim, ))
    #e1
    encoded = BatchNormalization()(input_layer)
    encoded = Dense(input_dim, activation='relu')(encoded)
    #e2
    encoded = BatchNormalization()(encoded)
    encoded = Dense(input_dim, activation='relu')(encoded)
    #e3
    encoded = BatchNormalization()(encoded)
    encoded = Dense(input_dim, activation='relu')(encoded)
    #e4
    encoded = BatchNormalization()(encoded)
    z_mean = enc_mean(encoded)
    z_log_var = enc_log_var(encoded)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    #D3
    decoded = BatchNormalization()(z)
    decoded = Dense(input_dim, activation='relu')(decoded)
    #D2
    decoded = BatchNormalization()(decoded)
    decoded = Dense(input_dim, activation='relu')(decoded)
    #D1
    decoded = BatchNormalization()(decoded)
    decoded = Dense(input_dim, activation='relu')(decoded)
    #output
    decoded = BatchNormalization()(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    #model
    vae = Model(input_layer, output_layer)
    vae_e = Model(input_layer, z_mean)

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
        callbacks=my_callbacks)

    # Use encoder part of the autoencoder for feature selection
    encoded_features_train = vae.predict(x_train)
    encoded_features_test = vae.predict(x_test)
    dim_fs = encoded_features_test.shape[1]

    #特徵選取時間結束
    end1 = time.process_time()
    tt1 = end1 - start1


    #存特徵選取後的檔案
    x_train_ae_df = pd.DataFrame(encoded_features_train)
    x_test_ae_df = pd.DataFrame(encoded_features_test)
    y_ans_train = pd.DataFrame(y_train, columns = ["Class"])
    y_ans_test = pd.DataFrame(y_test, columns = ["Class"])
    df1 = x_train_ae_df.merge(y_ans_train, how='left', left_index=True, right_index=True)
    df2 = x_test_ae_df.merge(y_ans_test, how='left', left_index=True, right_index=True)
    # df1.to_csv('/content/drive/My Drive/Colab Notebooks/output/' + dataset + '_vae_train_' + str(times) + '.csv', index=False, encoding='utf-8')
    # df2.to_csv('/content/drive/My Drive/Colab Notebooks/output/' + dataset + '_vae_test_' + str(times) + '.csv', index=False, encoding='utf-8')


    #存各項數值
    f_dim.append(dim_fs); f_time.append(tt1)

    #分類器時間開始
    start2 = time.process_time()

    #分類器
    svm = SVC(kernel='linear')
    svm.fit(encoded_features_train, y_train)
    y_predict = svm.predict(encoded_features_test)

    #標籤轉換
    y_predict_m = label_binarize(y_predict, classes=unique_y)
    y_test_m = label_binarize(y_test, classes=unique_y)

    #分數
    acc = accuracy_score(y_test_m, y_predict_m)
    auc = roc_auc_score(y_test_m, y_predict_m,multi_class="ovo")

    #分類器時間結束
    end2 = time.process_time()
    tt2 = end2 - start2

    #存ANS檔案
    y_guess = pd.DataFrame(y_predict, columns = ["svm_guess"])
    y_ans = pd.DataFrame(y_test, columns = ["true"])
    svm_ans = y_ans.merge(y_guess, how='left', left_index=True, right_index=True)

    #存各項數值
    svm_acc.append(acc); svm_auc.append(auc); svm_time.append(tt2)

    start3 = time.process_time()

    #分類器
    knn = KNeighborsClassifier()
    knn.fit(encoded_features_train, y_train)
    y_predict = knn.predict(encoded_features_test)

    #標籤轉換
    y_predict_m = label_binarize(y_predict, classes=unique_y)
    y_test_m = label_binarize(y_test, classes=unique_y)

    #分數
    acc = accuracy_score(y_test_m, y_predict_m)
    auc = roc_auc_score(y_test_m, y_predict_m, multi_class="ovo")

    end3 = time.process_time()
    tt3 = end3 - start3
    total_time = total_time + (end3 - start1)

    #存檔案
    y_guess = pd.DataFrame(y_predict, columns = ["knn_guess"])
    y_ans = pd.DataFrame(y_test, columns = ["true"])
    knn_ans = y_ans.merge(y_guess, how='left', left_index=True, right_index=True)
    test_ans = pd.merge(svm_ans, knn_ans, on=['true'], how='left')
    test_ans.to_csv('/content/drive/My Drive/Colab Notebooks/output/' + dataset + '_predict_vae_' + str(times) + '.csv', index=False, encoding='utf-8')

    #存各項數值
    knn_acc.append(acc);  knn_auc.append(auc); knn_time.append(tt3)

  #存結果檔案
  new = pd.DataFrame({'dataset': dataset,
              # 'svm_acc': np.mean(svm_acc),
              'svm_auc': np.mean(svm_auc),
              # 'svm_time': np.mean(svm_time),
              # 'knn_acc': np.mean(knn_acc),
              'knn_auc': np.mean(knn_auc),
              # 'knn_time': np.mean(knn_time),
              # 'f_dim': np.mean(f_dim),
              'ae_time': np.mean(f_time),
              'total_time': total_time
              }, index=[idx])

  result = result.append(new)

result.to_csv('/content/drive/My Drive/Colab Notebooks/output/(ans)vae_' + vae_version + '.csv', index=False, encoding='utf-8')
