#import libraries

#basics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from IPython.display import display, HTML

#stats
import math, time, random, datetime

#visualizing missing values
#import missingno as msno

#processing data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

#spliting and testing data
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn import metrics, model_selection, tree, preprocessing, linear_model
from sklearn.metrics import accuracy_score

#machine learning models
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

# model tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

#ignore warnings
#import warnings
#warnings.filterwarnings('ignore')

class Model:

    SEED = 42

    def __init__(self,data,spliter='simple',scaler=None,ycol=None,test_size=None,train_size=None):
        
        #get test_size and train_size
        if not (test_size and train_size):
            self.test_size,self.train_size = 0.2,0.8
        elif test_size:
            self.test_size = test_size
            self.train_size = 1-test_size
        else:
            self.test_size = 1-train_size
            self.train_size = train_size

        #get data (separating x and y)
        if ycol:
            self.x,self.y = data.drop(ycol,axis=1),data[ycol]

        #spliting data
        self.x_train,self.x_test,self.y_train,self.y_test = self.split(type=spliter)

        #scaling data
        if scaler:
            self.scale(type=scaler)

        #store model in variable
        self.model = None
        
    def return_data(self):
        return self.x_train,self.x_test,self.y_train,self.y_test
    
    def split(self,type='simple'):
        if type == 'simple':
            return train_test_split(self.x,self.y,
                        test_size=self.test_size,train_size=self.train_size,random_state=Model.SEED)
        else:
            pass

    def scale(self,type='Standard'):
        if type == 'Standard':
            scaler = StandardScaler().fit(self.x_train)
            self.x_train = scaler.transform(self.x_train)
            self.x_test = scaler.transform(self.x_test)

    def simple_acc(self,model=None):
            if not model:
                model = self.model
            #simple
            acc_train = round(model.score(self.x_train,self.y_train)*100,2)
            acc_test = round(model.score(self.x_test,self.y_test)*100,2)
            print("Accuracy Train: %s" % acc_train)
            print("Accuracy Test: %s" % acc_test)

    def cross_val_acc(self,algo,n_folds=10):
            #cross validation
            acc_cv = cross_val_score(algo,self.x,self.y,cv=n_folds)
            print("Accuracy CV 10-Fold: %s" % acc_cv)

    def apply_SVR(self,save=False,accs='simple',n_folds=10,scoring='neg_mean_squared_error'):
        algo = SVR()
        model = algo.fit(self.x_train,self.y_train)
        if save:
            self.model = model
        if accs=='simple':
            self.simple_acc(model)
        elif accs=='cross_val':
            self.cross_val_acc(algo,n_folds=n_folds)
        elif accs=='all':
            self.simple_acc(model)
            self.cross_val_acc(algo,n_folds=n_folds)

        return model
    
    def apply_model(self,algo=SVR(),save=False,accs='simple',n_folds=10,scoring='neg_mean_squared_error'):
        model = algo.fit(self.x_trian,self.y_train)
        if save:
            self.model = model
        if accs=='simple':
            self.simple_acc(model)
        elif accs=='cross_val':
            self.cross_val_acc(algo,n_folds=n_folds)
        elif accs=='all':
            self.simple_acc(model)
            self.cross_val_acc(algo,n_folds=n_folds)

        return model
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# Function that runs the requested algorithm and return accuracy metrics

def fit_ml_algo(algo,x_train,y_train,x_test,y_test,cv):
    
    start_time = time.time()
    
    #create model
    model = algo.fit(x_train,y_train)
    acc_train = round(model.score(x_train,y_train)*100,2)
    acc_test = round(model.score(x_test,y_test)*100,2)
    
    #cross validation
    train_pred = model_selection.cross_val_predict(algo,x_train,y_train,cv=_cv)
    
    #cross-validation accuracy metric
    acc_cv = round(metrics.accuracy_score(y_train,train_pred)*100,2)
    
    log_time = (time.time()-start_time)
    
    total_time = datetime.timedelta(seconds=log_time)
    
    #show results
    #print("Accuracy: %s" % acc_train)
    #print("Accuracy CV 10-Fold: %s" % acc_cv)
    #print("Running Time: %s" % datetime.timedelta(seconds=log_time))
    
    return train_pred, acc_train, acc_cv, acc_test, total_time, model

#applies an algorithm
def apply_algo(model,x_train,y_train,x_test,y_test,cv):
    train_pred , acc_train, acc_cv, acc_test, total_time, model = fit_ml_algo(model,x_train,y_train,x_test,y_test,cv)
                                                                            

    results = {
        "Model" : str(model).split('(')[0],
        "Accuracy Train" : acc_train,
        "Accuracy CV" : acc_cv,
        "Accuracy Test" : acc_test,
        "Running Time" : total_time
    }
    return results