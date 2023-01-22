import pickle
from glob import glob
import time
import optuna
import warnings
from sklearn.ensemble import IsolationForest
from pyod.models.iforest import IForest
from sklearn import model_selection
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dtime
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,mean_squared_error,make_scorer,r2_score,mean_absolute_error,recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor,MLPClassifier
from lightgbm import LGBMRegressor,LGBMClassifier
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor,SGDClassifier
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.dummy import DummyRegressor,DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from optuna.integration import OptunaSearchCV
from tqdm import tqdm
from IPython.display import clear_output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
simplefilter("ignore", category=ConvergenceWarning)

data_list= glob('../../data/processed/*') 

data1 =  pd.read_pickle(data_list[1])  #onehot encoded data
data1 = data1.dropna()

data2 =  pd.read_pickle(data_list[2])  #not encoded data
data2 = data2.dropna()

data1= data1.drop(['gun_sayisi_class'],axis=1)
data2= data2.drop(['gun_sayisi_class'],axis=1)


y1 = data1['gun_sayisi']
X1 = data1.drop(['gun_sayisi'],axis=1)


iforest = IForest()
outlier_labels = iforest.fit_predict(data1)
X1 = X1[outlier_labels==0]
y1= y1[outlier_labels==0]


qs = QuantileTransformer()
X1_scaled = X1.copy()
X1_scaled[X1.select_dtypes(['int64','float64']).columns] = qs.fit_transform(X1.select_dtypes(['int64','float64']))  #sayisal degerler oranlanir


estimators = {'Linear Regression':LinearRegression(),
              'XGBoost Regression':xgb.XGBRegressor(),
              'RandomForest Regression':RandomForestRegressor(),
              'AdaBoost Regression':AdaBoostRegressor(),
              'GradientBoosting Regression':GradientBoostingRegressor(),
              'LGBMRegression':LGBMRegressor(),
              'ElasticNet':ElasticNet(),
              'SGDRegressor':SGDRegressor(),
              'BayesianRidge':BayesianRidge(),
              'CatBoostRegressor':CatBoostRegressor(silent=True),
              'MevcutDurum_modeli':DummyRegressor(strategy='constant',constant=30),
}

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, shuffle=True,test_size=0.33, random_state=42)


def fit_score(models,X_train,X_test,y_train,y_test):
    np.random.seed(42)
    scores={}
    scores2={}
    for mname , model in models.items():
        model.fit(X_train,y_train)
        scores.update({mname:model.score(X_test,y_test)})
        y_pred = model.predict(X_test)
        y_pred = y_pred.clip(min=0)
        scores2.update({mname:mean_squared_error(y_pred,y_test)})
        print(mname,mean_absolute_error(y_pred,y_test))
    return scores2

preliminary_results= fit_score(estimators,X1_train,X1_test,y1_train,y1_test)