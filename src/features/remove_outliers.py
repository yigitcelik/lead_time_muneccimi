from glob import glob
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
import optuna
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")

"""
from ChatGPT:

Scikit-learn (sklearn) is a popular Python library for 
machine learning, and it provides several methods for 
detecting outliers in multidimensional data. Some of these methods are:

->Isolation Forest: This method uses random forests to isolate outliers. 
It assigns a score to each data point based on the number of splits 
required to isolate that point. Points with a low score are considered 
to be outliers.
->Local Outlier Factor (LOF): This method measures the local density of 
data points. Points that have a lower density compared to their neighbors 
are considered to be outliers.
->One-Class SVM: This method trains a support vector machine on 
only the inliers, and then uses the trained model to identify outliers.
->Elliptic Envelope: This method fits a Gaussian distribution to the data, 
and identifies points that are far away from the mean as outliers.
"""

data_list= glob('../../data/processed/*') 

data_not_encoded = pd.read_pickle("../../data/processed/data_processed_not_encoded.pickle")
data_not_encoded=data_not_encoded[~data_not_encoded["gun_sayisi_class"].isnull()] #sınıflandırmaya uymamış na değerleri eler.
data_encoded = pd.read_pickle("../../data/processed/data_processed_onehot_encoded.pickle")
data_encoded=data_encoded[~data_encoded["gun_sayisi_class"].isnull()] #sınıflandırmaya uymamış na değerleri eler.

num_cols=data_not_encoded.select_dtypes(["int","float"]).columns #numerik kolonları bulur
om =[LocalOutlierFactor(),OneClassSVM(),IsolationForest(),EllipticEnvelope()] #outlier bulma methodları

def find_outliers(df,method):
    """Gönderilen dataframe'deki satırlardan hangileri outlier ise onları yine 
    girdi olarak aldığı methodu
    kullanarak tespit eder.

    Args:
        df (dataframe): outlier satırların hesaplanacağı dataframe girdisi
        method (function): outlier bulma fonksiyonu

    Returns:
        Series: ilgili satır outlier ise True ,değilse False olan bir pandas 
        series döndürür.
    """
    df = df[num_cols]
    temp_array = method.fit_predict(df.values)
    df[str(method)]=temp_array
    df[str(method)]= df[str(method)].map({1:False,-1:True})
    return df[str(method)]

for met in om: #methodları ve dataframeleri göndererek outlier kolonlarında durumları kaydeder.
    data_not_encoded[str(met)+"_outlier"]=find_outliers(data_not_encoded,met)
    data_encoded[str(met)+"_outlier"]=find_outliers(data_encoded,met)



def fit_score(df):
    """tüm outlier bulma methodlarına göre dataseti ayarlayarak 
    RandomForest regressor kullanır ve mean_squared_error'ları hesaplar.
    çıkan sonuçları dataframe olarak döndürür.

    Args:
        df (dataframe): outlier methodlarının test edilmesi istenen dataframe

    Returns:
        dataframe: outlier methodu, onun mse score'u ve veri büyüklüğü değerlerini döndürür.
    """
    test_results={'method':[],'score':[],'size':[]}
    methods = [col for col in df.columns if "_outlier" in col]
    methods.append("No Method_outlier") #method olmadanki durumu karşılaştırmak için eklenir
    df["No Method_outlier"] =False #method olmayanlar için tüm değerler False yani outlier olmayan şeklinde kayıtlanır.
    for met in methods:
        df_filtered = df[df[met]==False] #outlier olmayanlar seçilir.
        y = df_filtered["gun_sayisi"] #target olarak y değişkenine atanır
        X = df_filtered.drop(["gun_sayisi_class","gun_sayisi"],axis=1) #targetler girdiden drop edilir.
        X = X.drop(methods,axis=1) #girdiden outlier method kolonları drop edilir.
        cv_score = cross_val_score(RandomForestRegressor(),X=X,y=y,cv=5,scoring="neg_mean_squared_error",verbose=0)
        test_results["method"].append(met)
        test_results["score"].append(np.abs(np.mean(cv_score)))
        test_results["size"].append(len(y))
    return pd.DataFrame(test_results)


fit_score(data_encoded).to_csv('../../reports/data_encoded_outlier_test_results.csv')
fit_score(data_not_encoded).to_csv('../../reports/data_not_encoded_outlier_test_results.csv')


#Isolation Forest ile data'yı gunceller ve diger gereksiz kolonları kaldırır.


def clean_filter(df):
    df = df[df["IsolationForest()_outlier"]==False]
    methods = [col for col in df.columns if "_outlier" in col]
    df =df.drop(methods,axis=1)
    return df


data_encoded = clean_filter(data_encoded)
data_not_encoded = clean_filter(data_not_encoded)

data_encoded.to_pickle("../../data/processed/data_processed_onehot_encoded_wo_outliers.pickle")
data_not_encoded.to_pickle("../../data/processed/data_processed_not_encoded_wo_outliers.pickle")
