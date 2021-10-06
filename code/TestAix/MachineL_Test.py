# on Cmd:
# pip install pydotplus
# pip install -U scikit-learn
# pip install ipython
# pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
# pip install -U matplotlib
# pip install seaborn

from math import gamma
from operator import mod
from numpy.core.numeric import cross
import pydotplus
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn import tree
from IPython.display import Image, display

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load libraries
#Visualize the data
from pandas import read_csv
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1.,1.]]
y = [0, 1]
clf = MLPClassifier(solver= 'lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5,2), random_state=1)
clf.fit(X,y)
MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1, solver= 'lbfgs')

clf.predict([[2., 2.], [-1., -2.]])

# Load dataset
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = read_csv(url, names=names)

# tag = [ 'Slept well yesterday?', 'Yesterday was stressful?' ]
# columns = [4,5]
# datset = read_csv(path, names=tag, usecols= columns, header=1)
# path = r"C:\Users\Aixzyl\Documents\Python\diabetes.csv"
# dataset1 = pd.read_csv(path)
# X=dataset1.iloc[:,0:8]
# y=dataset1.iloc[:,8]


# print(X.shape)
# print(y.shape)

# print(X.head())
# Split-out validation dataset
# array = datset.values
# X = array[:,0:6]
# y = array[:,1]
# X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.25, random_state=0) #Test_size uses percentage of the database

# Fit the model on training set
# model1 = LogisticRegression(max_iter=200)
# model1.fit(X_train, Y_train)

# model2 = DecisionTreeClassifier()
# model2.fit(X_train, Y_train)

# model3 = KNeighborsClassifier()
# model3.fit(X_train, Y_train)

# model4 = RandomForestClassifier()
# model4.fit(X_train, Y_train)

# y_predicted = model4.predict(X_validation)

# cm = confusion_matrix(Y_validation, y_predicted)

# print(cm)

# # Confusion Metrix using Heat Maps 

# class_names=[0,1] # name  of classes
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)
# # create heatmap
# sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
# ax.xaxis.set_label_position("top")
# plt.tight_layout()
# plt.title('Confusion matrix', y=1.1)
# plt.ylabel('Actual label')
# plt.xlabel('Predicted label') #First Two digits are accurate predicitons the bottom two are not

# logreg = LogisticRegression

# print("Accuracy:", metrics.accuracy_score(Y_validation, y_predicted))
# print("Precision:",metrics.precision_score(Y_validation, y_predicted))
# print("Recall:",metrics.recall_score(Y_validation, y_predicted))
# y_pred_proba = logreg.predict_proba(X_validation)[::,1]
# fpr, tpr, _ = metrics.roc_curve(Y_validation,  y_pred_proba)
# auc = metrics.roc_auc_score(Y_validation, y_pred_proba)
# plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
# plt.legend(loc=4)
# plt.show()

# # Spot Check Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr'))) # Logistic Regression (LR)
# models.append(('LDA', LinearDiscriminantAnalysis())) # Linear Discriminant Analysis (LDA)
# models.append(('KNN', KNeighborsClassifier())) # K-Nearest Neighbors (KNN).
# models.append(('CART', DecisionTreeClassifier())) # Classification and Regression Trees (CART).
# models.append(('NB', GaussianNB())) # Gaussian Naive Bayes (NB).
# models.append(('SVM', SVC(gamma='auto'))) # Support Vector Machines (SVM).



# Make predictions on validation dataset
# model = SVC(gamma='auto')
# model.fit(X_train, Y_train)
# predictions1 = model1.predict(X_validation)
# predictions2 = model2.predict(X_validation)
# predictions3 = model3.predict(X_validation)
# predictions4 = model4.predict(X_validation)

# Evaluate Predictions
# print('Logistic Regression = ' + str(accuracy_score(Y_validation, predictions1)))
# print('Decision Tree = ' + str(accuracy_score(Y_validation, predictions2)))
# print('KNeighborsClassifier = ' + str(accuracy_score(Y_validation, predictions3)))
# print('RandomForest = ' + str(accuracy_score(Y_validation, predictions3)))

# print(confusion_matrix(Y_validation, predictions2))
# print(classification_report(Y_validation, predictions2))



# # #Evaluate each model in turn
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# #Compare Algorithms
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()

# # Scatter plot matrix
# scatter_matrix(dataset)
# pyplot.show()
#
#
#

# #histograms
# dataset.hist()
# pyplot.show()

# # Box and Whisker plots
# dataset.plot(kind='box', subplots =True, layout=(2,2), sharex=False, sharey=False)
# pyplot.show()

# # shape
# print(dataset.shape)
# # head
# print(dataset.head(20))
# # descriptions
# print(dataset.describe())
# # class distribution
# print(dataset.groupby('class').size())
