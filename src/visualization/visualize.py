import pandas as pd
import numpy as np
from glob import glob
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, '/Users/yigitcelik/Desktop/github/gazi_bitirme/lead_time_muneccimi/')
from src.data.make_dataset import dataset

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Tur_t kolonu 1:seri uretim,2:pilot uretim,3:prototip uretim, 4: az sayida uretim
#Mip sorumlusu_t kolonu : 1: Diger,2: Mikrodalga malzemeler




data_list = glob('../../data/interim/*') 

data =  pd.read_pickle(data_list[0])

dataset.explore_df(data)

cat_cols = [col for col in data.select_dtypes('category').columns]
num_cols = [col for col in data.columns if data[col].dtype in ["int","float"]]

px.box(data[num_cols])

for col in cat_cols:
    fig = px.histogram(data[col],title=col,color=data[col])
    fig.show()
