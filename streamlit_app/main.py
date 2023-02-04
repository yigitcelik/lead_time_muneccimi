import streamlit as st
import pandas as pd
from create_dummy_sap_data import create_sap_data
from make_prediction import get_data
import time

st.set_page_config(page_title="Prediction Model", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title("Lead Time Muneccimi v 1.0")

#uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

col1,col2,col3 = st.columns(3)
wo_input = col1.text_input(label='Is emri No (coklu girdi yapilabilir) :')


if st.button("Tahmin Yap"):
    try:
        assert wo_input!=''
        wo_list = wo_input.split(' ')
        df = create_sap_data.create(wo_list)
        with st.spinner("SAP'den veri cekiliyor lutfen bekleyiniz"):
            time.sleep(2)
        st.success('Veri Cekimi ve tahminleme tamamlandi')
        st.write(get_data.get_predict(df))
    except:
        st.error('Veri Cekiminde ve Tahminlemede hata alindi')
    



