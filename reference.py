# https://ithelp.ithome.com.tw/articles/10304494
## import慣例
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
## Common imports
import sys
import sklearn # scikit-learn
import os
import scipy
## plot 視覺化
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

## 分割資料
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

## 報表結果、模型評估
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

## modelbuilding 各模型套件
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



files = ['sonar']
ansc45knn = pd.DataFrame()

for file in files :

  df = pd.read_csv('drive/My Drive/Colab Notebooks/datasets/' + file + '.csv')

  labelencoder = LabelEncoder()
  df['Class'] = labelencoder.fit_transform(df['Class'])

  y = df['Class']
  x = df.drop(['Class'], axis=1)
  x_c = df.drop(['Class'], axis=1)#取得columns名稱
  yy=np.unique(y)#多分類AUC轉換使用

  #資料正規化
  # minmax = preprocessing.MinMaxScaler()
  # x = minmax.fit_transform(x)
  # x=pd.DataFrame(x,columns = x_c.columns)


  #切割訓練集與測試集
  from sklearn.model_selection import train_test_split
  x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2 , random_state=42)
  # print('Training data shape:',x_train.shape)
  # print('Testing data shape:',x_test.shape)


  # Spot Check Algorithms
  models = []
  models.append(('LDA', LinearDiscriminantAnalysis()))
  models.append(('KNN', KNeighborsClassifier()))
  models.append(('CART', DecisionTreeClassifier()))
  models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=1)))
  models.append(('SVM', SVC(gamma='auto')))
  # evaluate each model in turn
  results = []
  names = []
  for name, model in models:
    kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

    #使用 cv ，並用validation data 計算 validation error 判斷哪種模型比較好
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# Compare Algorithms
# plt.boxplot(results, labels=names)
# plt.title('Algorithm Comparison')
# plt.show()
print()

## SVM classifier
model = SVC(kernel='linear')
model.fit(x_train, y_train)
predictions = model.predict(x_test)

# Evaluate predictions
acc = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, predictions, multi_class="ovo") # Area Under Curve
print('acc:', acc)
print('auc:', auc)
print(confusion_matrix(y_test, predictions)) # 混淆矩陣confusion matrix (tn, fp, fn, tp)
print(classification_report(y_test, predictions))
print()

model = KNeighborsClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
acc = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, predictions, multi_class="ovo") # Area Under Curve
print('acc:', acc)
print('auc:', auc)
print(confusion_matrix(y_test, predictions)) # 混淆矩陣confusion matrix (tn, fp, fn, tp)
print(classification_report(y_test, predictions))
print()
