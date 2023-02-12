import pandas as pd
import numpy as np
from glob import glob
from sklearn.preprocessing import OneHotEncoder
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
"""
High Cardinality olan kolonları bulup , bunların eşsiz değerlerinin önemsizlerini birleşirerek
cardinality'yi azaltmaya çalışır.

ikinci hem olduğu haliyle veya kategorik değişkenleri one hot encoded ederek kaydeder.

"""

data_list = glob('../../data/interim/*') 
data =  pd.read_pickle("../../data/interim/tumveri_transformed.pickle") #'../../data/interim/tumveri_transformed.pickle'

def car2cat(data):
    """Bir Dataframe'deki kategorik kolonlari inceler.Eger kolonlarda 12'den fazla degisik veri var ise
    bunlari cardinal olarak degerlendirip bu kolonlardaki verilerin toplama gore yuzdelerini kontrol eder.
    buyukten kucege dogru default sirali bu yuzdeleri toplar ve %95 degerine ulastigi zaman o kolondaki
    kalan degerlere(yani yuzdesel toplamlari maksimum %5 olan) 999 degerini atar.Bu sekilde cardinal
    olan verilerin essiz deger sayisini dusurmeye calisir.

    Args:
        data (dataframe): incelenecek dataframe

    Returns:
        dataframe,dictionary: cardinality'si azaltilmis yeni dataframe ile bu kolonlari ve yeni essiz degerlerini
        iceren dictionary'yi gonderir.
    """

    high_card_cat = {'name':[],'reduced_values':[]}
    for col in data.select_dtypes('category').columns:
        if data[col].nunique()>12:
            high_card_cat['name'].append(col)
            for i in range(1,data[col].nunique()):
                if data[col].value_counts(normalize=True)[0:i].sum()>0.95:
                    print(f"{col} kolonu deger sayisi {data[col].nunique()}'den {i+1}' ye dusuruldu")
                    data[col] = data[col].astype('int')
                    data.loc[~data[col].isin(data[col].value_counts()[0:i].index),col]=999
                    data[col] = data[col].astype('category')
                    high_card_cat['reduced_values'].append(data[col].value_counts().index.values)
                    break
    return data,high_card_cat


data,high_card_cat = car2cat(data)

high_card_cat= pd.DataFrame(high_card_cat)

high_card_cat.to_pickle('../../data/processed/highcardinal_columns.pickle')



def make_class(target):
    """aldiği kolondaki(pandas Series) verileri verilen bin değerlerine göre
    sınıflandırır.

    Args:
        target (Series): Pandas Series, sınıflandırılması istenen
        kolon.

    Returns:
        Series: Sınıflandırdığı verileri içeren pandas series.
    """
    bins=[0,7,14,25,50,50000]
    return pd.cut(target,bins=[0,7,14,25,50,2000],labels=[1,2,3,4,5])

data['gun_sayisi_class'] = make_class(data['gun_sayisi'])




data.to_pickle('../../data/processed/data_processed_not_encoded.pickle')



#one-hot encoding

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

for col in high_card_cat['name'].values:

    data=one_encode(col,high_card_cat.loc[high_card_cat['name']==col,'reduced_values'].values[0].to_numpy(),data)

data =one_encode('ISDT tarihinde eksiksiz mi ?',np.array([0,1]),data)
data =one_encode('PP02 adım var mı?',np.array([0,1]),data)
data = one_encode('Sipariş önceliği',np.array([0,1]),data)
data = one_encode('Türü_t',np.array([1,2,3,4]),data)
data = one_encode('MİP sorumlusu_t',np.array([1,2]),data)
data = one_encode('Hesap Tayin Tipi',np.array([0,1]),data)
data = one_encode('ISDT_Quarter',np.array([1,2,3,4]),data)
data = one_encode('ISDT_Month',np.arange(1,13),data)
data = one_encode('Plan Durumu',np.array([5,6,7,8]),data)

data.to_pickle('../../data/processed/data_processed_onehot_encoded.pickle')