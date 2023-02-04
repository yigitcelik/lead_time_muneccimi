import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,mean_squared_error,make_scorer,r2_score,mean_absolute_error,recall_score
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


class get_data():
    def get_predict(df):
        #make dataset
        data =  df.copy()
        value= 60 #60 essiz degiskenden az deger var ise numerik yap 
        cat_columns = [col for col in data[data.select_dtypes(['int','float']).columns].columns if data[col].nunique()<value]
        num_columns = [col for col in data[data.select_dtypes(['int','float']).columns].columns if col not in cat_columns]
        data[cat_columns] = data[cat_columns].astype('category')
        try:
            data = data.drop(['Plan Durumu','kaç üretim alanı ?'],axis=1)
        except:
            pass

        #build features
        card_col= pd.read_pickle("streamlit_app/highcardinal_columns.pickle")

        def set_cat(x,col):
            if x in (card_col.loc[card_col["name"]==col,"reduced_values"].to_list()[0]):
                return x
            else:
                return 999

        for col in card_col["name"].values:
            data[col]=data[col].apply(set_cat,args=[col])

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

        #prediction

        scaler = pickle.load(open('streamlit_app/neural_network(mae 9.1)_scaler.pkl', 'rb'))
        X = scaler.transform(data.drop(['is_emri'],axis=1))

        model =  pickle.load(open('streamlit_app/neural_network(mae 9.1).pkl', 'rb'))

        y_pred = model.predict(X).reshape(-1,)

        data['tahmini_lead_time(gun)'] = y_pred
        data['tahmini_lead_time(gun)'] = data['tahmini_lead_time(gun)'].astype(int)
        col = data.pop('tahmini_lead_time(gun)')
        data.insert(0,'tahmini_lead_time(gun)' , col)

        return data
        
