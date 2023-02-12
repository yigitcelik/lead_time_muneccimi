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
from sklearn.manifold import TSNE
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression,PoissonRegressor
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,mean_squared_error,make_scorer,r2_score,mean_absolute_error,recall_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
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

data1 =  pd.read_pickle('../../data/processed/data_processed_onehot_encoded_wo_outliers.pickle')  #onehot encoded data
data1 = data1.dropna()

data2 =  pd.read_pickle('../../data/processed/data_processed_not_encoded_wo_outliers.pickle')  #not encoded data
data2 = data2.dropna()

data1= data1.drop(['gun_sayisi_class'],axis=1)
test1 = data1.sample(1000)
data1 = data1[~data1.index.isin(test1.index)]

data1.to_pickle('../../data/processed/data_processed_onehot_encoded_wo_outliers_train-test.pickle')
test1.to_pickle('../../data/processed/data_processed_onehot_encoded_wo_outliers_validation.pickle')

data2= data2.drop(['gun_sayisi_class'],axis=1)
test2 = data2.sample(1000)
data2 = data2[~data2.index.isin(test2.index)]

data2.to_pickle('../../data/processed/data_processed_not_encoded_wo_outliers_train-test.pickle')
test2.to_pickle('../../data/processed/data_processed_not_encoded_wo_outliers_validation.pickle')

#what are the difference between quantiletransformer and minmaxscaler
def first_training(df,target,scaler,cat):
    y1 = df[target]
    X1 = df.drop([target],axis=1)

    if scaler:
        qs = QuantileTransformer()
        X1[X1.select_dtypes(['int64','float64']).columns] = qs.fit_transform(X1.select_dtypes(['int64','float64']))  #sayisal degerler oranlanir


    estimators = {
                'SVR':SVR(),
                'Linear Regression':LinearRegression(),
                'XGBoost Regression':xgb.XGBRegressor(),
                'RandomForest Regression':RandomForestRegressor(),
                #'AdaBoost Regression':AdaBoostRegressor(),
                'GradientBoosting Regression':GradientBoostingRegressor(),
                'LGBMRegression':LGBMRegressor(),
                'ElasticNet':ElasticNet(),
                #'SGDRegressor':SGDRegressor(),
                'BayesianRidge':BayesianRidge(),
                'CatBoostRegressor':CatBoostRegressor(silent=True),
                'PoissonRegressor':PoissonRegressor(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'KNeighborsRegressor':KNeighborsRegressor(),
                'MevcutDurum_modeli':DummyRegressor(strategy='constant',constant=30),
    }
    estimators2 = {
                'SVR':SVR(),
                'RandomForest Regression':RandomForestRegressor(),
                'GradientBoosting Regression':GradientBoostingRegressor(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'KNeighborsRegressor':KNeighborsRegressor(),
                'MevcutDurum_modeli':DummyRegressor(strategy='constant',constant=30),
    }


    def fit_score(models,X,y):
        np.random.seed(42)
        scores={}
        scores2={}
        for mname , model in tqdm(models.items()):
            cv_score =cross_val_score(model, X, y, cv=4,scoring="neg_mean_absolute_error",verbose=0)
            scores.update({mname:np.abs(np.mean(cv_score))})
            clear_output(wait=True)

        return scores
    if cat==0:
        preliminary_results = fit_score(estimators,X1,y1)
    elif cat==1:
        preliminary_results = fit_score(estimators2,X1,y1)

    return preliminary_results



results1= first_training(data1,'gun_sayisi',scaler=False,cat=0)
results2= first_training(data2,'gun_sayisi',scaler=False,cat=1)

results3= first_training(data1,'gun_sayisi',scaler=True,cat=0)
results4= first_training(data2,'gun_sayisi',scaler=True,cat=1)

fig1 = px.bar(x=results1.keys(),y=results1.values(),color=results1.keys(),title="One-hot encoded data ile ilk training sonuclari")
fig1.write_html("../../reports/figures/first_training_onehot.html")
fig2 = px.bar(x=results2.keys(),y=results2.values(),color=results2.keys(),title="Encode edilmemis data ile ilk training sonuclari")
fig2.write_html("../../reports/figures/first_training_not_encoded.html")
fig3 = px.bar(x=results3.keys(),y=results3.values(),color=results3.keys(),title="One-hot encoded Scaled data ile ilk training sonuclari")
fig3.write_html("../../reports/figures/first_training_onehot_scaled.html")
fig4 = px.bar(x=results4.keys(),y=results4.values(),color=results4.keys(),title="Encode edilmemis Scaled data ile ilk training sonuclari")
fig4.write_html("../../reports/figures/first_training_not_encoded_scaled.html")

y = data2["gun_sayisi"]
X = data2.drop("gun_sayisi",axis=1)


qs = QuantileTransformer()
X[X.select_dtypes(['int64','float64']).columns] = qs.fit_transform(X.select_dtypes(['int64','float64']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

def objective(trial):
    # Define the hyperparameters to optimize
    # n_estimators = trial.suggest_int('n_estimators', 50, 500)
    # learning_rate = trial.suggest_uniform('learning_rate', 0.01, 0.5)
    # max_depth = trial.suggest_int('max_depth', 1, 10)
    svr_c = trial.suggest_loguniform('C', 1e-4, 10)
    svr_eps = trial.suggest_loguniform('epsilon', 1e-4, 1)
    svr_kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    svr_degree = trial.suggest_int('degree', 2, 5)

    model = SVR(C=svr_c, epsilon=svr_eps, kernel=svr_kernel, degree=svr_degree)


    # Train the Gradient Boosting Regressor
    #model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=0)

    score =cross_val_score(model, X, y, cv=4,scoring="neg_mean_absolute_error",verbose=0)

    mae = np.abs(np.mean(score))


    # Use pruning to avoid time-consuming trials
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return mae

# Run the optimization
study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())

with tqdm(total=100) as progress_bar:
    for i in range(10):
        study.optimize(objective, n_trials=1)
        progress_bar.update(1)


# Get the best hyperparameters
best_params = study.best_params
best_score = study.best_value
print("Best hyperparameters:", best_params)
print("Best score:", best_score)


model =  SVR(**best_params)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred[y_pred<=0] = 1
mean_absolute_error(y_pred=y_pred,y_true=y_test)


#validation data score 
y_valid = test2["gun_sayisi"]
X_valid = test2.drop(["gun_sayisi"],axis=1)
X_valid[X_valid.select_dtypes(['int64','float64']).columns] = qs.transform(X_valid.select_dtypes(['int64','float64']))
y_predv =model.predict(X_valid)
y_predv[y_predv<=0] = 1
mean_absolute_error(y_pred=y_predv,y_true=y_valid)

#validation data score with dummy regressor
model_dummy = DummyRegressor(strategy='constant',constant=30)
model_dummy.fit(X_train,y_train)
mean_absolute_error(y_true=y_valid,y_pred=model_dummy.predict(X_valid))

def reg_scatter(y_pred,y_test):
    acc_ind = np.abs(y_pred-y_test)<=10

    plot_df =pd.DataFrame({'Gercek lead time':y_test,'Tahmin Lead time':y_pred,'Fark 10 gün icinde':acc_ind})

    fig3 = px.scatter(data_frame=plot_df,x="Gercek lead time",y="Tahmin Lead time",color="Fark 10 gün icinde")

    category_count = plot_df.groupby('Fark 10 gün icinde').size().reset_index(name='count')

    title = category_count['Fark 10 gün icinde'].astype(str) + " = " + category_count['count'].astype(str)
    title = ", ".join(title)

    fig3.update_layout(
        title=title,
    )
    return fig3


reg_scatter(y_predv,y_valid).write_html("../../reports/figures/svr_val_data.html")
reg_scatter(y_pred,y_test).write_html("../../reports/figures/svr_traintest_data.html")


def class_accuracy(y_pred,y_test):
    acc_ind = np.abs(y_pred-y_test)<=10
    plot_df =pd.DataFrame({'Gercek lead time':y_test,'Tahmin Lead time':y_pred,'Fark 10 gün icinde':acc_ind})
    plot_df["Gercek sınıf"]= pd.cut(plot_df["Gercek lead time"],bins=[0,15,30,2000],labels=['erken','orta','gec'])
    plot_df["Tahmin sınıf"]= pd.cut(plot_df["Tahmin Lead time"],[0,15,30,2000],labels=['erken','orta','gec'])
    return plot_df



plot_df = class_accuracy(y_predv,y_valid)
report = classification_report(y_true=plot_df["Gercek sınıf"],y_pred=plot_df["Tahmin sınıf"],output_dict=True)
report_df = pd.DataFrame(report).transpose()
cm =confusion_matrix(y_true=plot_df["Gercek sınıf"],y_pred=plot_df["Tahmin sınıf"],labels=['erken','orta','gec'])

sns.heatmap(report_df.iloc[:, :-1], annot=True)
plt.savefig("../../reports/figures/svr_classreport0-15-30.png")

pickle.dump(model, open('../../models/svr(mae 9.16).pkl', 'wb'))