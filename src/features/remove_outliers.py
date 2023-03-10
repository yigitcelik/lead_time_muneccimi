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
import sys
sys.path.insert(0, '/Users/yigitcelik/Desktop/github/lead_time_muneccimi/')
from src.visualization.visualize import tell_data


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")
""" Explanation of the Code
This code is importing several libraries and modules, 
including glob, sklearn, xgboost, pandas, numpy, scipy 
and optuna. It then sets some options for pandas to 
display dataframes. 

The code then uses the glob module to get a list of 
files from a directory. It then reads in two pickle 
files containing dataframes and removes any rows with
missing values in the gun_sayisi_class column. 

The code then creates a list of numerical columns 
from the dataframe and creates a list of outlier 
detection methods. It then defines a function that 
takes in a dataframe and an outlier detection method
as arguments and returns a series with True or False 
values depending on whether each row is an outlier or not. 
The code then loops through the list of outlier 
detection methods and applies them to both dataframes, 
creating new columns for each method with True or False 
values depending on whether each row is an outlier or 
not. 

The code then defines another function that takes 
in a dataframe as an argument and uses Random Forest 
Regressor to fit the model on the dataframe while 
calculating mean squared errors. The results are 
returned as a dataframe. The code then applies this 
function to both dataframes with outliers detected 
by all methods included in the list. 

Finally, the code defines another function that 
takes in a dataframe as an argument and filters 
it using Isolation Forest while dropping other 
unnecessary columns. This filtered version of the 
dataframes are saved as pickle files for future use.


"""


"""Outlier Detection Methods
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
data_not_encoded=data_not_encoded[~data_not_encoded["gun_sayisi_class"].isnull()] #s??n??fland??rmaya uymam???? na de??erleri eler.
data_encoded = pd.read_pickle("../../data/processed/data_processed_onehot_encoded.pickle")
data_encoded=data_encoded[~data_encoded["gun_sayisi_class"].isnull()] #s??n??fland??rmaya uymam???? na de??erleri eler.

num_cols=data_not_encoded.select_dtypes(["int","float"]).columns #numerik kolonlar?? bulur
om =[LocalOutlierFactor(),OneClassSVM(),IsolationForest(),EllipticEnvelope()] #outlier bulma methodlar??

def find_outliers(df,method):
    """G??nderilen dataframe'deki sat??rlardan hangileri outlier ise onlar?? yine 
    girdi olarak ald?????? methodu
    kullanarak tespit eder.

    Args:
        df (dataframe): outlier sat??rlar??n hesaplanaca???? dataframe girdisi
        method (function): outlier bulma fonksiyonu

    Returns:
        Series: ilgili sat??r outlier ise True ,de??ilse False olan bir pandas 
        series d??nd??r??r.
    """
    df = df[num_cols]
    temp_array = method.fit_predict(df.values)
    df[str(method)]=temp_array
    df[str(method)]= df[str(method)].map({1:False,-1:True})
    return df[str(method)]

for met in om: #methodlar?? ve dataframeleri g??ndererek outlier kolonlar??nda durumlar?? kaydeder.
    data_not_encoded[str(met)+"_outlier"]=find_outliers(data_not_encoded,met)
    data_encoded[str(met)+"_outlier"]=find_outliers(data_encoded,met)



def fit_score(df):
    """t??m outlier bulma methodlar??na g??re dataseti ayarlayarak 
    RandomForest regressor kullan??r ve mean_squared_error'lar?? hesaplar.
    ????kan sonu??lar?? dataframe olarak d??nd??r??r.

    Args:
        df (dataframe): outlier methodlar??n??n test edilmesi istenen dataframe

    Returns:
        dataframe: outlier methodu, onun mse score'u ve veri b??y??kl?????? de??erlerini d??nd??r??r.
    """
    test_results={'method':[],'score':[],'size':[]}
    methods = [col for col in df.columns if "_outlier" in col]
    methods.append("No Method_outlier") #method olmadanki durumu kar????la??t??rmak i??in eklenir
    df["No Method_outlier"] =False #method olmayanlar i??in t??m de??erler False yani outlier olmayan ??eklinde kay??tlan??r.
    for met in methods:
        df_filtered = df[df[met]==False] #outlier olmayanlar se??ilir.
        y = df_filtered["gun_sayisi"] #target olarak y de??i??kenine atan??r
        X = df_filtered.drop(["gun_sayisi_class","gun_sayisi"],axis=1) #targetler girdiden drop edilir.
        X = X.drop(methods,axis=1) #girdiden outlier method kolonlar?? drop edilir.
        cv_score = cross_val_score(RandomForestRegressor(),X=X,y=y,cv=5,scoring="neg_mean_absolute_error",verbose=0)
        test_results["method"].append(met)
        test_results["score"].append(np.abs(np.mean(cv_score)))
        test_results["size"].append(len(y))
    return pd.DataFrame(test_results)


fit_score(data_encoded).to_csv('../../reports/data_encoded_outlier_test_results.csv')
fit_score(data_not_encoded).to_csv('../../reports/data_not_encoded_outlier_test_results.csv')


#Isolation Forest ile data'y?? gunceller ve diger gereksiz kolonlar?? kald??r??r.


def clean_filter(df):
    df = df[df["IsolationForest()_outlier"]==False]
    methods = [col for col in df.columns if "_outlier" in col]
    df =df.drop(methods,axis=1)
    return df


data_encoded = clean_filter(data_encoded)
data_not_encoded = clean_filter(data_not_encoded)

data_encoded.to_pickle("../../data/processed/data_processed_onehot_encoded_wo_outliers.pickle")
data_not_encoded.to_pickle("../../data/processed/data_processed_not_encoded_wo_outliers.pickle")
