# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from glob import glob
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



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





data,num_columns,cat_columns = dataset.df_dtype_creater('../../data/raw/live_data.xlsx',value=60)

#data tipleri kontrol edilir yanlisliklar duzeltilir.

#cat_columns = [cat_columns.remove(col) for col in ['kac adimli?','kaç farklı iş yeri var?','kaç üretim alanı ?','ıskarta sayısı']]
num_columns.extend(('kac adimli?','kaç farklı iş yeri var?','kaç üretim alanı ?','ıskarta sayısı'))
data[['kac adimli?','kaç farklı iş yeri var?','kaç üretim alanı ?','ıskarta sayısı']]= data[['kac adimli?','kaç farklı iş yeri var?','kaç üretim alanı ?','ıskarta sayısı']].astype('int')



#'Plan durumu' kolonu eksik veri iceriyor, 'kac uretim alani' kolonunda yanlis bilgiler mevcut bu sebeple bu kolonlar cikartilir

try:
    data = data.drop(['Plan Durumu','kaç üretim alanı ?'],axis=1)
except:
    pass

#duzenlenen dataframe pickle olarak kaydedilir.
data.to_pickle('../../data/interim/livedata_transformed.pickle')
