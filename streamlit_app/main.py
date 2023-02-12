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


uploaded_file = col1.file_uploader("Girdi dosyasini excel yukle", type=["xlsx", "xls"])


if st.button("Tahmin Yap"):
    if wo_input!="":
        wo_list = wo_input.split(' ')
        df = create_sap_data.create(wo_list)
        with st.spinner("SAP'den veri cekiliyor lutfen bekleyiniz"):
            time.sleep(2)
        col1.success('Veri Cekimi ve tahminleme tamamlandi')
        st.write(get_data.get_predict(df))
    elif uploaded_file is not None:
        wo_list = wo_input.split(' ')
        df = pd.read_excel(uploaded_file)
        with st.spinner("Tahmin yapiliyor lutfen bekleyiniz"):
            time.sleep(2)
        col1.success('Tahminleme tamamlandi')
        st.write(get_data.get_predict(df))
    else:
        col1.error("Girdi verilmedi !!!")
        

        

