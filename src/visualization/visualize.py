import pandas as pd
import numpy as np
from glob import glob
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy import stats
from scipy.stats import norm, skew
sys.path.insert(0, '/Users/yigitcelik/Desktop/github/lead_time_muneccimi/')
from src.data.make_dataset import dataset

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Tur_t kolonu 1:seri uretim,2:pilot uretim,3:prototip uretim, 4: az sayida uretim
#Mip sorumlusu_t kolonu, 1: Diger,2: Mikrodalga malzemeler
#Siparis onceligi kolonu, 1:yok 0:var
#ISDT tarihinde eksiksiz mi ?, 1:evet , 0:hayir
#PP02 adim var mi ?, 1:yok, 0:var


data_list= glob('../../data/interim/*') 

data =  pd.read_pickle(data_list[0])

def tell_data(data):
    dataset.explore_df(data)

    cat_cols = [col for col in data.select_dtypes('category').columns]
    num_cols = [col for col in data.columns if data[col].dtype in ["int","float"]]

    for col in num_cols:
        fig = px.box(data[col],title=col)
        fig.show()

    for col in cat_cols:
        fig = px.histogram(data[col],title=col,color=data[col])
        fig.update_xaxes(type='category')
        fig.show()

    def percent95(x):
        return x.quantile(0.95)
    def percent75(x):
        return x.quantile(0.75)


    print(data.groupby(['MİP sorumlusu_t']).agg({'mean','median',percent75,percent95,'max'}).T,'\n')
    print(data.groupby(['Türü_t']).agg({'mean','median',percent75,percent95,'max'}).T,'\n')

    numeric_feats = data.dtypes[data.dtypes != "category"].index
    skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna()))
    print(skewed_feats[skewed_feats > 0.75])


tell_data(data)
