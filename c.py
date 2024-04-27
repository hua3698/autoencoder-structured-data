from common import *
from library import *
from ae import *
from dae import *

datasets = ['page-blocks-1-3_vs_4', 'yeast-2_vs_8', 'kddcup-land_vs_portsweep']
version = 'ae_230'
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
    x_train_encoded, x_test_encoded = train_ae_230(x_train, x_test)

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
    x_train_smote_encoded, x_test_smote_encoded = train_ae_230(x_train_smote, x_test)

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
  
  result = pd.concat([result, new], ignore_index=True)
result.to_csv('result/(ans)'+ version + '_' + dataset + '.csv',index=False,encoding='utf-8')


version = 'ae_240'
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
    x_train_encoded, x_test_encoded = train_ae_240(x_train, x_test)

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
    x_train_smote_encoded, x_test_smote_encoded = train_ae_240(x_train_smote, x_test)

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
  
  result = pd.concat([result, new], ignore_index=True)
result.to_csv('result/(ans)'+ version + '_' + dataset + '.csv',index=False,encoding='utf-8')

version = 'dae_230'
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
    x_train_encoded, x_test_encoded = train_dae_230(x_train, x_test)

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
    x_train_smote_encoded, x_test_smote_encoded = train_dae_230(x_train_smote, x_test)

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
  
  result = pd.concat([result, new], ignore_index=True)
result.to_csv('result/(ans)'+ version + '_' + dataset + '.csv',index=False,encoding='utf-8')

version = 'dae_240'
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
    x_train_encoded, x_test_encoded = train_dae_240(x_train, x_test)

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
    x_train_smote_encoded, x_test_smote_encoded = train_dae_240(x_train_smote, x_test)

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
  
  result = pd.concat([result, new], ignore_index=True)
result.to_csv('result/(ans)'+ version + '_' + dataset + '.csv',index=False,encoding='utf-8')
