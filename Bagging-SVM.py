import pickle
import warnings
import os
from multiprocessing import Pool
import time
import sys
import random
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
# import more_itertools as mit
import csv
from collections import defaultdict
# import pywt
# import train as train
from scipy.signal import *
from scipy import *
from pylab import *
from sklearn import preprocessing, manifold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, cross_val_predict, KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, recall_score, \
    precision_score, matthews_corrcoef, auc
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


KF = KFold(n_splits=10, shuffle=True, random_state=0)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, roc_curve

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,100)

def SVM(file_x, file_y):
    X = np.genfromtxt(file_x, delimiter=',')
    y = np.genfromtxt(file_y)
    # print(y)
    X, y = np.array(X), np.array(y).astype(np.int)
    X = LinearDiscriminantAnalysis(n_components=1).fit(X, y).transform(X)
    # LDA
    ACCs = 0
    Pres = 0
    Recs = 0
    F1s = 0
    AUCs = 0
    i = 1
    for train_index, test_index in KF.split(X):
        print("第" + str(i) + "重交叉验证")
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        print(y_test)

        print("SVM model")
        svm_linear = SVC(C=1.0, kernel='linear', random_state=0, probability=True)
        svm_linear = BaggingClassifier(base_estimator=svm_linear,n_estimators=350)
        svm_linear.fit(x_train, y_train)
        # SVM 模型
        predict_y = svm_linear.predict(x_test)
        # print("测试集预测 : ", svm_linear.predict(test_x))
        # print("测试集准确度: ", svm_linear.score(x_test, y_test))
        y_pred = svm_linear.predict(x_test)
        # print(classification_report(y_test, y_pred))
        # print("SVM Result - ", end="")
        # 计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # interp:插值 把结果添加到tprs列表中
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # 计算auc
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (i, roc_auc))

        i = i + 1
        cm = confusion_matrix(y_test, predict_y).astype(np.int64)
        print(cm)
        print("ACC: %f " % accuracy_score(y_test, y_pred))
        print("F1: %f " % f1_score(y_test, y_pred))
        print("Recall: %f " % recall_score(y_test, y_pred))
        print("Pre: %f " % precision_score(y_test, y_pred))
        # print("MCC: %f " % matthews_corrcoef(y_test, y_pred))
        print("AUC: %f " % roc_auc_score(y_test, y_pred))
        ACCs = ACCs + accuracy_score(y_test, y_pred)
        F1s = F1s + f1_score(y_test, y_pred)
        Recs = Recs + recall_score(y_test, y_pred)
        Pres = Pres + precision_score(y_test, y_pred)
        AUCs = AUCs + roc_auc_score(y_test, y_pred)
    print("AVG-------------------------")
    print("ACC: %f" % (ACCs / 10))
    print("Pre: %f" % (Pres / 10))
    print("Recs: %f" % (Recs / 10))
    print("F1s: %f" % (F1s / 10))
    print("AUCs: %f" % (AUCs / 10))
    """
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Bagging-SVM-ROC')
    plt.legend(loc='lower right')
    plt.show()

    
    # 训练集测试集划分
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42)
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    # print("test_y: ");
    # print(test_y)
    # SVM Model
    print("SVM model")
    svm_linear = SVC(kernel='linear', random_state=0, probability=True)
    svm_linear.fit(train_x, train_y)

    # SVM 模型
    predict_y = svm_linear.predict(test_x)
    #print("测试集预测 : ", svm_linear.predict(test_x))
    print("测试集准确度: ", svm_linear.score(test_x, test_y))


    y_pred = svm_linear.predict(test_x)
    print(classification_report(test_y, y_pred))
    print("SVM Result - ", end="")

    # 混淆矩阵可视化
    cm = confusion_matrix(test_y, predict_y).astype(np.int64)
    # print(cm)

    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title('SVM_' + '_Confusion_Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Real Labels')
    plt.show()

    # ROC 曲线绘制
    probas_y = cross_val_predict(svm_linear, train_x, train_y, cv=10, method="predict_proba")
    # print(probas_y)
    scores_y = probas_y[:, 1]
    fpr, tpr, thresholds = roc_curve(train_y, scores_y)
    plt.plot(fpr, tpr, label="SVM")

    plt.plot([0, 1], [0, 1], 'k--', label="mid")
    plt.legend(loc="lower right")
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.title('SVM_' + '_ROC')
    plt.show()


    # 验证集
    file2_x = ".\\feature3.csv"
    file2_y = ".\\label3.csv"
    test2_X = np.genfromtxt(file2_x, delimiter=',')
    test2_y = np.genfromtxt(file2_y)
    test2_X, test2_y = np.array(test2_X), np.array(test2_y).astype(np.int)
    test2_X = LinearDiscriminantAnalysis(n_components=1).fit(test2_X, test2_y).transform(test2_X)
    predict2_y = svm_linear.predict(test2_X)
    # print("测试集2预测 : ", svm_linear.predict(test2_X))
    print("测试集2准确度: ", svm_linear.score(test2_X, test2_y))
    y_pred2 = svm_linear.predict(test2_X)
    print(classification_report(test2_y, y_pred2))
    print("SVM Result - ", end="")
    print(accuracy_score(test2_y, y_pred2))
    print(precision_score(test2_y, y_pred2))
    print(recall_score(test2_y, y_pred2))
    print(f1_score(test2_y, y_pred2))
    print(roc_auc_score(test2_y, y_pred2))

    # 混淆矩阵可视化
    cm2 = confusion_matrix(test2_y, predict2_y).astype(np.int64)
    print(cm2)
    sns.heatmap(cm2, annot=True, cmap='Blues')
    plt.title('SVM_' + '_Confusion_Matrix2')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Real Labels')
    plt.show()
"""


SVM(".\\feature1.csv", ".\\label1.csv")

