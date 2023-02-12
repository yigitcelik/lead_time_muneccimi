# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from glob import glob
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
data_list = glob('../../data/raw/*') #raw data klasorundeki tum dosyalarin adreslerini kaydeder
#okunan dosya '../../data/raw/tum_data_v2.xlsx'
class dataset:
    

    def df_dtype_creater(file,value):
        """Numerik veri tiplerinin icerisinde yer alan kategorik
        tipteki kolonlari, eger bir kolonda value'dan daha az essiz veri varsa 
        bulur ve cat_columns'a ekler. ve  buldugu kategorik degiskenleri
        numerikten kategorige cevirir.

        Args:
            file (string): veri kaynagi dosya yolu
            value (int): kategorik-numerik ayrimini belirleyen esik deger

        Returns:
            df,list,list: Dataframe,numerik kolonlar,kategorik kolonlar
        """

        data =  pd.read_excel(file,parse_dates=True)
        cat_columns = [col for col in data[data.select_dtypes(['int','float']).columns].columns if data[col].nunique()<value]
        num_columns = [col for col in data[data.select_dtypes(['int','float']).columns].columns if col not in cat_columns]
        data[cat_columns] = data[cat_columns].astype('category')
        return data,num_columns,cat_columns





    #veri tipleri,eksik datalar ve numerik kolonlar on kontrol edilir.
    def explore_df(df):
        print(df.info(),'\n')
        print(df.isnull().sum(),'\n')
        print(df.describe(percentiles=np.arange(0.2,1.1,0.2)).T,'\n')





data,num_columns,cat_columns = dataset.df_dtype_creater('../../data/raw/model_input(ilk veri).xlsx',value=60)

#data tipleri kontrol edilir yanlisliklar duzeltilir.

#cat_columns = [cat_columns.remove(col) for col in ['kac adimli?','kaç farklı iş yeri var?','kaç üretim alanı ?','ıskarta sayısı']]
num_columns.extend(('kac adimli?','kaç farklı iş yeri var?','kaç üretim alanı ?','ıskarta sayısı','Z4 bildirim sayısı'))
data[['kac adimli?','kaç farklı iş yeri var?','kaç üretim alanı ?','ıskarta sayısı','Z4 bildirim sayısı']]= data[['kac adimli?','kaç farklı iş yeri var?','kaç üretim alanı ?','ıskarta sayısı','Z4 bildirim sayısı']].astype('int')

dataset.explore_df(data)

data["Plan Durumu"]=data["Plan Durumu"].fillna(8)
data.loc[((data["Plan Durumu"]==4) | (data["Plan Durumu"]==3)),"Plan Durumu"]=8
data["Plan Durumu"] = data["Plan Durumu"].cat.remove_categories(3)
data["Plan Durumu"] = data["Plan Durumu"].cat.remove_categories(4)

#'Plan durumu' kolonu eksik veri iceriyor, 'kac uretim alani' kolonunda yanlis bilgiler mevcut bu sebeple bu kolonlar cikartilir


#data = data.drop(['Plan Durumu','kaç üretim alanı ?'],axis=1)

data = data.drop(['Plan Durumu'],axis=1)

obj_cols = data.select_dtypes('object').columns
for col in data[obj_cols]:
    data.loc[data[col]=="Hata",col]=999
data[obj_cols]=data[obj_cols].astype('category')

#duzenlenen dataframe pickle olarak kaydedilir.
data.to_pickle('../../data/interim/tumveri_transformed.pickle')
