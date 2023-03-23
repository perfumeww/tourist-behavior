#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:45:46 2022

@author: songsongchangjiang
"""

''' Classification for Q26 '''

'''Required Package'''
# Data Mining
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

#features selection for LogisticRegression and SVM
from mlxtend.feature_selection import SequentialFeatureSelector as SFS #Backward and Forward
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV

# f1 score and Matrix
from sklearn.metrics import (confusion_matrix,precision_score,recall_score,accuracy_score)
from sklearn.metrics import f1_score

# Train and Test Set Split
from sklearn.model_selection import train_test_split

# Data visualization
import matplotlib.pyplot as plt


'''Data import and Cleaning'''

df = pd.read_csv('Project_Data_Discriminant_N1.csv')

cols = df.columns
for col in cols:
    print(col)
    Na_num = df[col].isnull().sum()
    print(Na_num)
    # get a list of unique values
    unique = df[col].unique()

    # if number of unique values is less than 30, print the values. Otherwise print the number of unique values
    if len(unique)<30:
        print(unique, '\n====================================\n\n')
    else:
        print(str(len(unique)) + ' unique values', '\n====================================\n\n')
df_discriminant = df

df_discriminant['EDUCATION'] = df_discriminant['EDUCATION'].replace(['BACHELOR','HighSchool','Master','3YCollege','VSC'],[2,5,1,3,4])

df_discriminant['AGE_Bins'] = pd.cut(df_discriminant['AGE'], 5)
df_discriminant[['AGE_Bins', 'Q26']].groupby(['AGE_Bins'], as_index=False).mean().sort_values(by='AGE_Bins', ascending=True)

#df_discriminant.to_csv("Project_Data_Discriminant_N2.csv", index= True)

df_discriminant.loc[(df_discriminant['AGE'] >18) &(df_discriminant['AGE']<=34),'AGE'] =1
df_discriminant.loc[(df_discriminant['AGE'] >34) &(df_discriminant['AGE']<=47),'AGE'] =2
df_discriminant.loc[(df_discriminant['AGE'] >47) &(df_discriminant['AGE']<=60),'AGE'] =3
df_discriminant.loc[(df_discriminant['AGE'] >60) &(df_discriminant['AGE']<=74),'AGE'] =4
df_discriminant.loc[(df_discriminant['AGE'] >74) &(df_discriminant['AGE']<=88),'AGE'] =5


df = df.drop(columns= ['TRANSPORTATION','Unnamed: 0','PROFESSION','AGE_Bins'])


df_dummy = pd.get_dummies(df,drop_first=True)
#len(df_dummy.columns)
df_dummy.columns
x = df_dummy.drop(columns = ['Q26_Yes'])
y = df_dummy['Q26_Yes']

################################################################################
################################################################################
#Logistic Regression
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.3,random_state = 1)
logmodel = LogisticRegression(solver = 'liblinear')
logmodel.fit(x_train,y_train)

prediction = logmodel.predict(x_test)
probab = logmodel.predict_proba(x_test)

c_mat= pd.DataFrame(confusion_matrix(y_test,prediction),index = ["Actual:0","Actual:1"], columns = ['Pred:0','Pred:1'])
c_mat
print('accuracy:', accuracy_score(y_test,prediction)) #accuracy:  0.7954545454545454
print('precision:',precision_score(y_test,prediction)) #precision:  0.7857142857142857
print('recall:',recall_score(y_test,prediction)) #recall: 0.88
f1_score(y_test,prediction) #0.830188679245283
y_pred_train = logmodel.predict(x_train) #通过检查训练集，来观察是否存在overfitting 的问题
f1_score(y_train,y_pred_train) #0.8318584070796461， 答案是基本不存在的，两个几乎一样

accuracy_score(y_train,y_pred_train) #0.81
accuracy_score(y_test,prediction) # 0.7954545454545454 
# 在Accuracy rate中，训练集效果稍高于 test集

F1_Logistic_withoutFS= 0.830188679245283
Acc_score_Logistic_withoutFS = 0.7954545454545454


''' Model prediction with SequentialFeatureSelector method_Backward'''

logmodel = LogisticRegression(solver = 'liblinear')
sfs = SFS(logmodel, 
          k_features=(1,15), 
          forward=False, 
          scoring='neg_root_mean_squared_error', 
          cv=20) 
sfs.fit(x_train, y_train)
sfs.k_feature_names_

X_train_sfs = sfs.transform(x_train)
X_test_sfs = sfs.transform(x_test)

# Fit the model using the new feature subset
# and make a prediction on the test data
logmodel.fit(X_train_sfs, y_train)
prediction = logmodel.predict(X_test_sfs)
probab = logmodel.predict_proba(X_test_sfs)
logmodel.coef_

c_mat= pd.DataFrame(confusion_matrix(y_test,prediction),index = ["Actual:0","Actual:1"], columns = ['Pred:0','Pred:1'])
c_mat
print('accuracy:', accuracy_score(y_test,prediction)) #accuracy: 0.8181818181818182
print('precision:',precision_score(y_test,prediction)) #precision: 0.84
print('recall:',recall_score(y_test,prediction)) #0.84

y_pred_train = logmodel.predict(X_train_sfs) 
print('accuracy——train:', accuracy_score(y_train,y_pred_train)) #accuracy——train: 0.82
f1_score(y_train,y_pred_train) #0.8392857142857144
f1_score(y_test,prediction) #0.8399999999999999
# LOG model 有显著提升，且没有OF问题， 但是目前的 features只剩下了5个 ('EDUCATION', 'DISTANCE', 'ANXIETY', 'TRAVEL_DURATION', 'Q25_Yes')

F1_Logistic_SFS_Back= 0.8399999999999999
Acc_score_Logistic_SFS_Back= 0.8181818181818182


'''Data visualization'''

import matplotlib.pyplot as plt
X_test_sfs_df= pd.DataFrame(X_test_sfs)

X_test_sfs_df = X_test_sfs_df.rename(columns = {0:'EDUCATION',1:'DISTANCE',2:'ANXIETY',3:'TRAVEL_DURATION',4:'Q25_Yes'})
plt.rcParams['axes.unicode_minus']=False 
coef_LR = pd.Series(logmodel.coef_.flatten(),index = X_test_sfs_df.columns ,name = 'Var')
plt.figure(figsize=(8,8))
coef_LR.sort_values().plot(kind='barh')
plt.title("Variances Importances")

# Visualization                       

coef_log = pd.DataFrame({'var' : X_test_sfs_df.columns,
                        'coef' : logmodel.coef_.flatten()
                        })

index_sort =  np.abs(coef_log['coef']).sort_values(ascending = False).index
coef_log_sort = coef_log.loc[index_sort,:]

plt.figure(figsize=(14,8))
x, y = coef_log_sort['var'], coef_log_sort['coef']
rects = plt.bar(x, y, color='dodgerblue')
plt.grid(linestyle="-.", axis='y', alpha=0.4)
plt.tight_layout()
y1 = y[ y > 0];x1 = x[y1.index]
for a,b in zip(x1,y1):
    plt.text(a ,b+0.02,'%.2f' %b, ha='center',va='bottom',fontsize=12)
y2 = y[ y < 0];x2 = x[y2.index]
for a,b in zip(x2,y2):
    plt.text(a ,b-0.02,'%.2f' %b, ha='center',va='bottom',fontsize=12)

# Coefficient explain:



from sklearn.inspection import plot_partial_dependence, partial_dependence


partial_dependence(logmodel, X_train_sfs, [0] )

plot_partial_dependence(logmodel, X_train_sfs,[0, (0, 1)])



b=logmodel.coef_
a=logmodel.intercept_
b
Prob_Changes = np.exp(b)/(1+np.exp(b))
X_test_sfs


new_char_list_1 = [ (5, 3, 2, 2, 0) ]

y_pred_1 = logmodel.predict(new_char_list_1)
y_pred_1
# 上面这个是对的

#第一个 = 0， 0.86300736, 0.13699264


prediction 
y_pred_1
probab 

def Probability(x1,x2,x3,x4,x5):
    var1 = 0.66199984 -0.50642295*x1 -0.38436859*x2 -0.11353801*x3 +0.73490031*x4+ 0.95310122*x5
    var2 = np.exp(var1)
    var3 = 1+ np.exp(var1)
    var4 = var2/var3
    return var4
Probability(5, 3, 2, 2, 0) #0.1442418890694979  小于0.5 所以是 0，不去旅游
Probability(3, 2, 1, 3, 0) #0.6142367806086575  大于0.5 所以是 1，Yes
Probability(3, 2, 3, 5, 0) #0.8465612810289268  大于0.5 所以是 1 

Probability(4, 2, 3, 5, 0) #0.7687841195884931 
0.8465612810289268 - 0.7687841195884931 = 0.07777716144043367
Probability(5, 2, 3, 5, 0) # 0.667086145515258 
0.7687841195884931 - 0.667086145515258 = 0.10169797407323511

Probability(1,1,1,1,0)
Probability(2,1,1,1,0)


Probability(1,1,1,1,0) - Probability(2,1,1,1,0) #0.1253337470257796
Probability(2,1,1,1,0) - Probability(3,1,1,1,0) #0.12184877709614977
Probability(3,1,1,1,0) - Probability(4,1,1,1,0) #0.10494789956858652 
Probability(4,1,1,1,0) - Probability(5,1,1,1,0) #0.0813677051353108


'''
1. 其中marginal effects 会随着其他 features 的变化而产生不同的变化，没有固定的变化值
    x1 的变化可以减少 人们选择旅游的概率，相反增加人们不选择旅游的概率， 影响变化平均百分之10，为最大
    x2 和 人们选择旅游的概率成负相关，和人们拒绝旅游的概率成正相关 概率的影响 百分之9
    x3 和 人们选择旅游的概率成负相关，和人们拒绝旅游的概率成正相关 概率的影响最低每次百分之2
    x4 和 人们选择旅游的概率为正相关，和人民拒绝旅游的概率成负相关，概率的影响最高为百分之 15，随着系数的增加，变化概率逐渐减少到百分之3的变化
    x5 和 人们选择旅游的概率为正相关，和人民拒绝旅游的概率成负相关，概率的影响最高位百分之 20

2. 总结发现其实, 没有办法确定5个F 各自的边际效应，但是至少我们可以知道这每个feature和概率的相关性。
3. 出现以上无法判定 恒定边际效应的原因可能是因为 feature之间存在一定的联系，并不是完全独立的。

'''



for i in range(1,5):
    Probability_Changes_x1= Probability(i,1,1,1,0)-Probability(i+1,1,1,1,0)
    print(Probability_Changes_x1)

    #0.1253337470257796
    #0.12184877709614977
    #0.10494789956858652
    #0.0813677051353108

for i in range(1,5):
    Probability_Changes_x1= Probability(i,2,1,1,0)-Probability(i+1,2,1,1,0)
    print(Probability_Changes_x1)

0.12409164550769586
0.10992948802440572
0.08724861613743068
0.06340193982094684

for i in range(1,5):
    Probability_Changes_x1= Probability(i,3,1,1,0)-Probability(i+1,3,1,1,0)
    print(Probability_Changes_x1)

0.11441035084692541
0.09307599662034394
0.06893948917784215
0.04759487095485959
################################################
for i in range(1,5):
    Probability_Changes_x2= Probability(1,i,1,1,0)-Probability(1,i+1,1,1,0)
    print(Probability_Changes_x2)
    #0.09485087444602991
    #0.09499868794779265
    #0.08852171676667742
    #0.0771089235044431

for i in range(1,5):
    Probability_Changes_x3= Probability(1,1,i,1,0)-Probability(1,1,i+1,1,0)
    print(Probability_Changes_x3)
    #0.027592330901179074
    #0.028029561656528035
    #0.028293251604335157
    #0.028376717848154154

for i in range(1,5):
    Probability_Changes_x4= Probability(1,1,1,i,0)-Probability(1,1,1,i+1,0)
    print(Probability_Changes_x4)
    #-0.158470497101046
    #-0.11020123116162384
    #-0.06511267968708756
    #-0.03483038025123941

    Probability_Changes_x5= Probability(1,1,1,1,0)-Probability(1,1,1,1,1)
    print(Probability_Changes_x5) #-0.1965197574334986

    Probability_Changes_xn= Probability(1,1,1,2,0)-Probability(1,1,1,2,1)
    print(Probability_Changes_xn) #-0.13362864029542576
    
    Probability_Changes_xn1= Probability(1,1,1,3,0)-Probability(1,1,1,4,1)
    print(Probability_Changes_xn1) #-0.10651444823961909


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logmodel.predict(X_test_sfs))
fpr, tpr, thresholds = roc_curve(y_test, logmodel.predict_proba(X_test_sfs)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

''' Model prediction with SequentialFeatureSelector method_Forward'''

logmodel = LogisticRegression(solver = 'liblinear')
sfs = SFS(logmodel, 
          k_features=(1,15), 
          forward=True, 
          scoring='neg_root_mean_squared_error', #no need to worry about
          cv=20) 
sfs.fit(x_train, y_train)
print(sfs.k_feature_names_, len(sfs.k_feature_names_))
#('EDUCATION', 'DISTANCE', 'ACCOMMODATION', 'BUDGET', 'NECESSITY', 'COMPANION_3-5', 'Q25_Yes') 7

X_train_sfs = sfs.transform(x_train)
X_test_sfs = sfs.transform(x_test)

# Fit the model using the new feature subset
# and make a prediction on the test data
logmodel.fit(X_train_sfs, y_train)
prediction = logmodel.predict(X_test_sfs)
probab = logmodel.predict_proba(X_test_sfs)


c_mat= pd.DataFrame(confusion_matrix(y_test,prediction),index = ["Actual:0","Actual:1"], columns = ['Pred:0','Pred:1'])
c_mat
print('accuracy:', accuracy_score(y_test,prediction)) #accuracy:  0.7727272727272727
print('precision:',precision_score(y_test,prediction)) #precision:0.7777777777777778
print('recall:',recall_score(y_test,prediction)) # recall: 0.84

y_pred_train = logmodel.predict(X_train_sfs) 
print('accuracy_train:', accuracy_score(y_train,y_pred_train)) #0.81

f1_score(y_train,y_pred_train) #0.8318584070796461
f1_score(y_test,prediction) #0.8076923076923077
#  不合适，出现OF 问题 且 各方面score 下降

F1_Logistic_SFS_Forward= 0.8076923076923077
Acc_score_Logistic_SFS_Forward= 0.7727272727272727
''' Model prediction with ExhaustiveFeature'''

efs = EFS(logmodel, 
          min_features=1,
          max_features=15,
          scoring='neg_root_mean_squared_error',
          cv=10)
efs.fit(x_train, y_train)

##selected features
efs.best_feature_names_ #('EDUCATION', 'DISTANCE', 'BUDGET', 'NECESSITY', 'Q25_Yes')

X_train_efs = efs.transform(x_train)
X_test_efs = efs.transform(x_test)

# Fit the estimator using the new feature subset
# and make a prediction on the test data
logmodel.fit(X_train_efs, y_train)
y_pred = logmodel.predict(X_test_efs)
c_mat= pd.DataFrame(confusion_matrix(y_test,y_pred ),index = ["Actual:0","Actual:1"], columns = ['Pred:0','Pred:1'])
c_mat
print('accuracy:', accuracy_score(y_test,y_pred )) #accuracy:  0.7727272727272727
print('precision:',precision_score(y_test,y_pred )) #precision: 0.7777777777777778
print('recall:',recall_score(y_test,y_pred ))  # recall: 0.84
y_pred_train = logmodel.predict(X_train_efs) 
f1_score(y_train,y_pred_train) #0.8318584070796461
f1_score(y_test,y_pred) # 0.8076923076923077
# 效果一般
F1_Logistic_efs= 0.8076923076923077
Acc_score_Logistic_efs= 0.7727272727272727

'''The best k feature'''

 for i in range(1,15):
        bestfeatures =  SelectKBest(score_func = f_regression,k=i) ### initialize
        bestfeatures.fit(x_train,y_train) ### training(finding correlation between data)
        
        #bestfeatures.get_support()
        
        new_x_train = bestfeatures.transform(x_train)
        new_x_test =  bestfeatures.transform(x_test)
        
        ## building the model on selected columns 
        logmodel = LogisticRegression()
        logmodel.fit(new_x_train, y_train)
        
        predictions  = logmodel.predict(new_x_test)
        
        F1_test_after_skbest = f1_score(y_test, predictions)
        print("no of features:",i)
        print ("F1_test_after_skbest:", F1_test_after_skbest)
# When K =14， F1 hits to the Maxium

F1_Logistic_Kbest= 0.830188679245283
Acc_score_Logistic_Kbest= 0.7954545454545454


################################################################################
################################################################################
'''SVM'''
model = SVC()

param_grid = {'C':[1,10,100],'gamma' : [1,0.1,0.01],'kernel':['rbf','linear']}

grid = GridSearchCV(model,param_grid, verbose = 3, scoring = 'f1')
grid.fit(x_train,y_train)

# find the best parameters
grid.best_params_

### can the model improve using the tuned paramethers
model = SVC(C= 1, gamma=1, kernel = 'linear')
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

c_mat= pd.DataFrame(confusion_matrix(y_test,y_pred ),index = ["Actual:0","Actual:1"], columns = ['Pred:0','Pred:1'])
c_mat
print('accuracy:', accuracy_score(y_test,y_pred )) #accuracy:  0.7727272727272727
print('precision:',precision_score(y_test,y_pred )) #precision: 0.7586206896551724
print('recall:',recall_score(y_test,y_pred ))  # recall: 0.88
y_pred_train = model.predict(x_train) 
f1_score(y_train,y_pred_train) #0.8421052631578948
f1_score(y_test,y_pred) # 0.8148148148148148

F1_SVM= 0.8148148148148148
Acc_score_SVM= 0.7727272727272727

# 有轻微的 OF 问题，但综合不如Logistic Model 效果好
################################################################################
################################################################################
'''Decision Tree'''
dt = DecisionTreeClassifier()
parameter_grid = {'max_depth':range(1,15),'min_samples_split':range(2,10)}

grid = GridSearchCV(dt,parameter_grid,verbose =3, scoring = 'f1',cv = 20) #initialize
grid.fit(x_train,y_train)

#best params
grid.best_params_

### tree using the tuned parameters

dt = DecisionTreeClassifier(max_depth=3 , min_samples_split= 2, random_state=1)
dt.fit(x_train, y_train)
dt_pred_test = dt.predict(x_test)
c_mat= pd.DataFrame(confusion_matrix(y_test,dt_pred_test),index = ["Actual:0","Actual:1"], columns = ['Pred:0','Pred:1'])
c_mat
print('accuracy:', accuracy_score(y_test,dt_pred_test )) #accuracy:  0.6590909090909091
print('precision:',precision_score(y_test,dt_pred_test )) #precision: 0.631578947368421
print('recall:',recall_score(y_test,dt_pred_test )) #recall: 0.96

## on training data using f1 score
dt_pred_train = dt.predict(x_train)
f1_score(y_train,dt_pred_train) #0.8091603053435115

dt_pred_test = dt.predict(x_test)
f1_score(y_test,dt_pred_test) #0.7619047619047619

F1_Dt= 0.7619047619047619
Acc_score_Dt= 0.6590909090909091
# Overfitting 而且效果不好

################################################################################
################################################################################
'''Random Forest '''
rfc = RandomForestClassifier()

parameter_grid = {'max_depth':range(1,18),'min_samples_split':range(2,10),'n_estimators':[50,100,150,200]}

grid = GridSearchCV(rfc,parameter_grid,verbose =3, scoring = 'f1',cv = 5)
grid.fit(x_train,y_train)

grid.best_params_

rfc = RandomForestClassifier(random_state =1, n_estimators = 200, max_depth= 1, min_samples_split= 8)
rfc.fit(x_train,y_train)
rfc_pred_test = rfc.predict(x_test)

c_mat= pd.DataFrame(confusion_matrix(y_test,rfc_pred_test),index = ["Actual:0","Actual:1"], columns = ['Pred:0','Pred:1'])
c_mat
print('accuracy:', accuracy_score(y_test,rfc_pred_test )) #accuracy:  0.8181818181818182
print('precision:',precision_score(y_test,rfc_pred_test )) #precision: 0.7741935483870968
print('recall:',recall_score(y_test,rfc_pred_test )) #recall: 0.96

## on train data using f1 score

rfc_pred_train = rfc.predict(x_train)
f1_score(y_train, rfc_pred_train) # 0.803030303030303

## testing data

rfc_pred_test = rfc.predict(x_test)
f1_score(y_test,rfc_pred_test) #0.8571428571428571
# 没有OF 的问题，而且是目前为止 F1 score 最高的

F1_rfc= 0.8571428571428571
Acc_score_rfc= 0.8181818181818182
################################################################################
################################################################################
'''KNeighbors'''

knn = KNeighborsClassifier()
param_grid = {'n_neighbors': range(1,50),'p':[1,2]}

grid = GridSearchCV(knn,param_grid,verbose= 3,scoring ='f1',cv =20)
grid.fit(x_train,y_train)

grid.best_params_ #{'n_neighbors': 19, 'p': 1}

knn = KNeighborsClassifier(n_neighbors=19,p=1)
knn.fit(x_train, y_train)
knn_pred = knn.predict(x_test)

c_mat= pd.DataFrame(confusion_matrix(y_test,knn_pred),index = ["Actual:0","Actual:1"], columns = ['Pred:0','Pred:1'])
c_mat
print('accuracy:', accuracy_score(y_test,knn_pred)) #accuracy:  0.7954545454545454
print('precision:',precision_score(y_test,knn_pred)) #precision: 0.7857142857142857
print('recall:',recall_score(y_test,knn_pred)) #recall:  0.88

knn_pred_train = knn.predict(x_train) 
f1_score(y_train, knn_pred_train) #0.7812500000000001

knn_pred = knn.predict(x_test)
f1_score(y_test, knn_pred) #0.830188679245283
# 还行 但有点怪
F1_Knn= 0.830188679245283
Acc_score_Knn=  0.7954545454545454
################################################################################
################################################################################
'''Gaussian Naive Bayes'''

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
prediction = gaussian.predict(x_test)

c_mat= pd.DataFrame(confusion_matrix(y_test,prediction),index = ["Actual:0","Actual:1"], columns = ['Pred:0','Pred:1'])
c_mat
print('accuracy:', accuracy_score(y_test,prediction)) #accuracy: 0.75
print('precision:',precision_score(y_test,prediction)) #precision: 0.7692307692307693
print('recall:',recall_score(y_test,prediction)) #recall: 0.8

y_pred_train = gaussian.predict(x_train) 
f1_score(y_train,y_pred_train) #0.7850467289719626
f1_score(y_test,y_pred) #0.8148148148148148

F1_gaussian= 0.8148148148148148
Acc_score_gaussian=  0.75
################################################################################
################################################################################
'''Perceptron'''

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
prediction = perceptron.predict(x_test)

c_mat= pd.DataFrame(confusion_matrix(y_test,prediction),index = ["Actual:0","Actual:1"], columns = ['Pred:0','Pred:1'])
c_mat
print('accuracy:', accuracy_score(y_test,prediction))# accuracy: 0.4318181818181818
print('precision:',precision_score(y_test,prediction)) #precision: Nan
print('recall:',recall_score(y_test,prediction)) # recall: 0

y_pred_train = perceptron.predict(x_train) 
f1_score(y_train,y_pred_train) #0.712871287128713
f1_score(y_test,y_pred) #0
# 不行
################################################################################
################################################################################
'''Stochastic Gradient Descent'''

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
prediction = sgd.predict(x_test)

c_mat= pd.DataFrame(confusion_matrix(y_test,prediction),index = ["Actual:0","Actual:1"], columns = ['Pred:0','Pred:1'])
c_mat
print('accuracy:', accuracy_score(y_test,prediction))# accuracy: 0.7727272727272727
print('precision:',precision_score(y_test,prediction)) #precision:  0.7272727272727273
print('recall:',recall_score(y_test,prediction)) # recall: 0.96
y_pred_train = sgd.predict(x_train) 
f1_score(y_train,y_pred_train) #0.8091603053435115
f1_score(y_test,y_pred) #0.8148148148148148


F1_sgd= 0.8148148148148148
Acc_sgd=  0.7727272727272727
################################################################################
################################################################################
'''LinearSVC'''
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
prediction= linear_svc.predict(x_test)

c_mat= pd.DataFrame(confusion_matrix(y_test,prediction),index = ["Actual:0","Actual:1"], columns = ['Pred:0','Pred:1'])
c_mat
print('accuracy:', accuracy_score(y_test,prediction))# accuracy: 0.7954545454545454
print('precision:',precision_score(y_test,prediction)) #precision:  0.7857142857142857
print('recall:',recall_score(y_test,prediction)) # recall: 0.88
y_pred_train = linear_svc.predict(x_train) 
f1_score(y_train,y_pred_train) #0.8214285714285715
f1_score(y_test,y_pred) #0.8148148148148148

F1_linear_svc= 0.8148148148148148
Acc_linear_svc=  0.7954545454545454
################################################################################
################################################################################
'''Voting  CV =10'''
lg = LogisticRegression(solver = 'liblinear')
rfc = RandomForestClassifier(random_state =1, n_estimators = 200, max_depth= 1, min_samples_split= 8)
sgd = SGDClassifier()
Svm = SVC(C= 1, gamma=1, kernel = 'linear')
model_combo = VotingClassifier(estimators = [ ('lrp',lg),('rfc',rfc),('sv',Svm),('sgd',sgd)],voting= 'hard')
model_combo_score = cross_val_score(model_combo,x,y,scoring = 'f1',cv =10, verbose =3)
print ('model combo', model_combo_score.mean()) # 0.780030525030525
# f1 不高 对比之前 单独的

lg = LogisticRegression(solver = 'liblinear')
rfc = RandomForestClassifier(random_state =1, n_estimators = 200, max_depth= 1, min_samples_split= 8)
sgd = SGDClassifier()
Svm = SVC(C= 1, gamma=1, kernel = 'linear')
model_combo = VotingClassifier(estimators = [ ('lrp',lg),('rfc',rfc),('sv',Svm),('sgd',sgd)],voting= 'hard')
model_combo_score = cross_val_score(model_combo,x,y,scoring = 'accuracy',cv =10, verbose =3)
print ('model combo', model_combo_score.mean()) #0.7442857142857142

F1_Voting= 0.780030525030525
Acc_Voting=  0.7442857142857142
################################################################################
################################################################################

F1_Logistic_withoutFS= 0.830188679245283
Acc_score_Logistic_withoutFS = 0.7954545454545454

F1_Logistic_SFS_Back= 0.8399999999999999
Acc_score_Logistic_SFS_Back= 0.8181818181818182

F1_Logistic_SFS_Forward= 0.8076923076923077
Acc_score_Logistic_SFS_Forward= 0.7727272727272727

F1_Logistic_efs= 0.8076923076923077
Acc_score_Logistic_efs= 0.7727272727272727

F1_Logistic_Kbest= 0.830188679245283
Acc_score_Logistic_Kbest= 0.7954545454545454

F1_SVM= 0.8148148148148148
Acc_score_SVM= 0.7727272727272727

F1_Dt= 0.7619047619047619
Acc_score_Dt= 0.6590909090909091

F1_rfc= 0.8571428571428571
Acc_score_rfc= 0.8181818181818182

F1_Knn= 0.830188679245283
Acc_score_Knn=  0.7954545454545454

F1_gaussian= 0.8148148148148148
Acc_score_gaussian=  0.75

F1_sgd= 0.8148148148148148
Acc_sgd=  0.7727272727272727

F1_linear_svc= 0.8148148148148148
Acc_linear_svc=  0.7954545454545454

F1_Voting= 0.780030525030525
Acc_Voting=  0.7442857142857142


DF_ML  = pd.DataFrame({
        'Model':['Support Vector Machines', 'KNN', 'Logistic Regression withoutFS', 'Logistic Regression SFS backward' , 
                  'Logistic Regression SFS forward', 'Logistic Regression EFS', 'Logistic Regression Kbest',
                  'Random Forest', 'Naive Bayes',
                  'Stochastic Gradient Decent', 'Linear SVC','Voting',
                  'Decision Tree'],
        'F1_Score': [F1_SVM, F1_Knn, F1_Logistic_withoutFS, F1_Logistic_SFS_Back,
                  F1_Logistic_SFS_Forward,F1_Logistic_efs, F1_Logistic_Kbest, 
                  F1_rfc, F1_gaussian, F1_sgd, F1_linear_svc,F1_Voting,F1_Dt],
        'Acc_score': [Acc_score_SVM, Acc_score_Knn, Acc_score_Logistic_withoutFS, Acc_score_Logistic_SFS_Back
                      ,Acc_score_Logistic_SFS_Forward,Acc_score_Logistic_efs, Acc_score_Logistic_Kbest,
                      Acc_score_rfc,Acc_score_gaussian,Acc_sgd,Acc_linear_svc,Acc_Voting,Acc_score_Dt]})

################################################################################
################################################################################
#Appendix
'''
 ## default KNN

logmodel = LogisticRegression(solver = 'liblinear')
Dt = DecisionTreeClassifier()
Svm = SVC()
knn = KNeighborsClassifier() ## default KNN
model_combo = VotingClassifier(estimators = [ ('lrp',logmodel),('dt',Dt),('sv',Svm)],voting= 'hard')

logmodel_score = cross_val_score(logmodel,x,y,scoring = 'f1',cv =10, verbose =10)
print ('logmodel', logmodel_score.mean()) # 0.75515873015873

model_log_predict = cross_val_predict(logmodel,x,y,cv =10, verbose =3)
model_log_predict
f1_score(y,model_log_predict)

Dt_score = cross_val_score(Dt,x,y,scoring = 'f1',cv =10, verbose =3)
print ('Dt', Dt_score.mean()) #0.6915196078431372

Svm_score = cross_val_score(Svm,x,y,scoring = 'f1',cv =10, verbose =3)
print ('Svm', Svm_score.mean()) #0.6365537843798714

Knn_score = cross_val_score(knn,x,y,scoring = 'f1', cv = 10, verbose =3)
print ('knn', Knn_score.mean()) #0.7149109387344682

model_combo_score = cross_val_score(model_combo,x,y,scoring = 'f1',cv =10, verbose =3)
print ('model combo', model_combo_score.mean()) #0.7325228512457614

model_combo_predict = cross_val_predict(model_combo,x,y,cv =10, verbose =3)
model_combo_predict
len(model_combo_predict)
f1_score(y,model_combo_predict)
pred = list([23, 5,	4,3	,5,2,4	,2	,0,	0,	0,	0,	0	,0	,0,	1,	0,	0	,1,	0,	0,	1,	0,	0,	1,	0,	1])
model_combo.predict(pred)'''





