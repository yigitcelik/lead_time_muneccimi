import pandas as pd
import numpy as np
from glob import glob
from sklearn.preprocessing import OneHotEncoder
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
"""
not encoded ve one hot encoded olarak 2 pickle dosyası oluşturur.
"""


data = pd.read_pickle('../../data/interim/livedata_transformed.pickle')

card_col= pd.read_pickle("../../data/processed/highcardinal_columns.pickle")

def set_cat(x,col):
    """High Cardinal olduğu bilinen kolonları önceden belirlenmis
    değerler ile map'leyerek daha az eşsiz değere sahip olmasını
    sağlar.

    Args:
        x (int): Kolondaki değerler
        col (string): kolon adı

    Returns:
        _type_: _description_
    """
    if x in (card_col.loc[card_col["name"]==col,"reduced_values"].to_list()[0]):
        return x
    else:
        return 999

for col in card_col["name"].values:
    data[col]=data[col].apply(set_cat,args=[col])



data.to_pickle('../../data/processed/data_processed_not_encoded_for_live.pickle')


def one_encode(col,array,df):
    """dataseti, kolon adını ve değerleri içeren array'i alır ve
    onehotencode yöntemi ile değerleri 0,1 olarak encode'layarak
    yeni kolonlar yaratır. yarattığı bu kolonları gönderilen 
    dataset'e ekler. Array verilmesinin sebebi her defasında aynı 
    kolon adları ile aynı sırada encoded kolonlarının eklenmesini
    sağlamaktır.

    Args:
        col (string): kolon adı
        array (numpy array): encode'lanacak değerleri içeren array
        df (dataframe): pandas dataframe

    Returns:
        dataframe: encode'lanmis dataframe'i döndürür.
    """
    oe_temp = OneHotEncoder(handle_unknown='ignore',sparse=False)
    oe_temp.fit(array.reshape(-1,1))
    temp_ = oe_temp.transform(df[col].astype('int32').values.reshape(-1,1))
    temp_ = pd.DataFrame(temp_,columns=oe_temp.get_feature_names())
    temp_= temp_.add_prefix(col)
    df = pd.concat([df.reset_index(),temp_.reset_index()],axis=1)
    df = df.drop([col],axis=1)
    try:
        df = df.drop(['level_0','index'],axis=1)
    except:
        df = df.drop(['index'],axis=1)
    return df

for col in card_col['name'].values:

    data=one_encode(col,card_col.loc[card_col['name']==col,'reduced_values'].values[0].to_numpy(),data)

data =one_encode('ISDT tarihinde eksiksiz mi ?',np.array([0,1]),data)
data =one_encode('PP02 adım var mı?',np.array([0,1]),data)
data = one_encode('Sipariş önceliği',np.array([0,1]),data)
data = one_encode('Türü_t',np.array([1,2,3,4]),data)
data = one_encode('MİP sorumlusu_t',np.array([1,2]),data)
data = one_encode('Hesap Tayin Tipi',np.array([0,1]),data)
data = one_encode('ISDT_Quarter',np.array([1,2,3,4]),data)
data = one_encode('ISDT_Month',np.arange(1,13),data)

data.to_pickle('../../data/processed/data_processed_onehot_encoded_for_live.pickle')