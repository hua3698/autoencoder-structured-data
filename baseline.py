from common import *
from library import *

# baseline

result = pd.DataFrame()
datasets = ['polish_year1', 'polish_year2', 'polish_year3', 'polish_year4', 'polish_year5']
# datasets = ['wisconsin', 'yeast1', 'haberman', 'vehicle1', 'page-blocks0', 'page-blocks-1-3_vs_4', 'yeast-2_vs_8', 'kddcup-land_vs_portsweep']

for idx, dataset in enumerate(datasets):

  minmax = preprocessing.MinMaxScaler()

  # svm_acc = [];  knn_acc = []
  svm_auc = [];  knn_auc = []; c45_auc = []; cart_auc = []
  smote_svm_auc = [];  smote_knn_auc = [];  smote_c45_auc = [];  smote_cart_auc = []

  for times in range(1,6):

    training = "{}_{}{}.csv".format(dataset, 'train', times)
    testing = "{}_{}{}.csv".format(dataset, 'test', times)

    df_train = pd.read_csv('../dataset/bank/' + dataset + '/' + training, delimiter=',')
    df_test = pd.read_csv('../dataset/bank/' + dataset + '/' + testing, delimiter=',')
    # df_train = pd.read_csv('../dataset/' + dataset + '-5-fold/' + training, delimiter=',')
    # df_test = pd.read_csv('../dataset/' + dataset + '-5-fold/' + testing, delimiter=',')

    # labelencoder = LabelEncoder()
    # y_train = labelencoder.fit_transform(df_train['Class'])
    # y_test = labelencoder.fit_transform(df_test['Class'])

    y_train = pd.DataFrame(df_train, columns = ['Class'])
    y_test = pd.DataFrame(df_test, columns = ['Class'])

    x_train = df_train.drop(['Class'], axis=1)
    x_test = df_test.drop(['Class'], axis=1)

    # string_columns = x_train.select_dtypes(include=['object']).columns
    # for col in string_columns:

    #     labelencoder = LabelEncoder()
    #     x_train[col] = labelencoder.fit_transform(df_train[col])
    #     x_test[col] = labelencoder.fit_transform(x_test[col])

    #特徵縮放
    # x_train_minmax = minmax.fit_transform(x_train)
    # x_test_minmax = minmax.fit_transform(x_test)

    # x_train = pd.DataFrame(x_train_minmax, columns = x_train.columns)
    # x_test = pd.DataFrame(x_test_minmax, columns = x_test.columns)

    acc, auc = train_orig_svc(x_train, x_test, y_train, y_test)
    svm_auc.append(auc)
    # svm_acc.append(acc);  svm_auc.append(auc)

    acc, auc = train_orig_knn(x_train, x_test, y_train, y_test)
    knn_auc.append(auc)
    # knn_acc.append(acc);  knn_auc.append(auc)

    auc = run_c45(x_train, x_test, y_train, y_test)
    c45_auc.append(auc)

    auc = run_cart(x_train, x_test, y_train, y_test)
    cart_auc.append(auc)


    x_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(x_train, y_train)
    x_train_smote = pd.DataFrame(x_train_smote, columns=x_train.columns)
    y_train_smote = pd.DataFrame(y_train_smote, columns=['Class'])

    acc, auc = train_orig_svc(x_train_smote, x_test, y_train_smote, y_test)
    smote_svm_auc.append(auc)

    acc, auc = train_orig_knn(x_train_smote, x_test, y_train_smote, y_test)
    smote_knn_auc.append(auc)

    auc = run_c45(x_train_smote, x_test, y_train_smote, y_test)
    smote_c45_auc.append(auc)

    auc = run_cart(x_train_smote, x_test, y_train_smote, y_test)
    smote_cart_auc.append(auc)

  print(dataset)
  print(np.mean(svm_auc))
  print(np.mean(knn_auc))
  print(np.mean(c45_auc))
  print(np.mean(cart_auc))
  print(np.mean(smote_svm_auc))
  print(np.mean(smote_knn_auc))
  print(np.mean(smote_c45_auc))
  print(np.mean(smote_cart_auc))
  print('------------------')

  time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  new = [dataset, svm_auc, knn_auc, c45_auc, cart_auc, smote_svm_auc, smote_knn_auc, smote_c45_auc, smote_cart_auc, time]

  with open('result/bank.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(new)

  # new=pd.DataFrame({'dataset':dataset,
  #           'svm_auc': np.mean(svm_auc),
  #           'knn_auc': np.mean(knn_auc),
  #           'c45_auc': np.mean(c45_auc),
  #           'cart_auc': np.mean(cart_auc),
  #           'smote_svm_auc': np.mean(smote_svm_auc),
  #           'smote_knn_auc': np.mean(smote_knn_auc),
  #           'smote_c45_auc': np.mean(smote_c45_auc),
  #           'smote_cart_auc': np.mean(smote_cart_auc),
  #           'now': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
  #           }, index=[idx])
  
  # result = pd.concat([result, new], ignore_index=True)
# result.to_csv('result/bank/.csv', index=False, encoding='utf-8')
