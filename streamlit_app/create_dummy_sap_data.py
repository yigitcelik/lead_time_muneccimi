import pandas as pd
import numpy as np

class create_sap_data():
    def create(wo_list):
        df_dict = {'is_emri':wo_list,
            'Hedeflenen Miktar': np.random.randint(1,101,len(wo_list)),
            'Hesap Tayin Tipi':np.random.randint(0,1,len(wo_list)),
            'Sipariş önceliği' : np.random.randint(0,1,len(wo_list)),
            'ISDT tarihinde eksiksiz mi ?' : np.random.randint(0,1,len(wo_list)),
            'kac adimli?' : np.random.randint(1,20,len(wo_list)),
            'kaç farklı iş yeri var?' : np.random.randint(1,12,len(wo_list)),
            #'kaç üretim alanı ?' : np.random.randint(1,4,len(wo_list)),
            #'PP02 adım var mı?' : np.random.randint(0,1,len(wo_list)),
            'toplam işçilik süresi(sa)' : np.random.randint(0,100,len(wo_list)),
            'toplam makine süresi(sa)' : np.random.randint(0,100,len(wo_list)),
            'Z4 bildirim sayısı' : np.random.randint(0,20,len(wo_list)),
            'Z6 bildirim sayısı' : np.random.randint(0,20,len(wo_list)),
            'ıskarta sayısı' : np.random.randint(0,4,len(wo_list)),
            'Türü_t' : np.random.randint(1,4,len(wo_list)),
            'Üy Srm_t' : np.random.randint(1,20,len(wo_list)),
            'mal grubu ilk 5 karakter_t' : np.random.randint(1,900,len(wo_list)),
            'MİP sorumlusu_t' : np.random.randint(0,1,len(wo_list)),
            'İlk İş Yeri_t' : np.random.randint(1,60,len(wo_list)),
            'ISDT_Quarter' : np.random.randint(1,4,len(wo_list)),
            'ISDT_Month' : np.random.randint(1,12,len(wo_list)),
            'Plan Durumu' : np.random.randint(5,8,len(wo_list)),
        }

        return pd.DataFrame(df_dict)
