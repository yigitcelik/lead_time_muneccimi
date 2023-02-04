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


data = pd.read_pickle("../../data/processed/data_processed_onehot_encoded_for_live.pickle")


scaler = pickle.load(open('../../models/neural_network(mae 9.1)_scaler.pkl', 'rb'))

X = scaler.transform(data)

model =  pickle.load(open('../../models/neural_network(mae 9.1).pkl', 'rb'))

y_pred = model.predict(X)

print(y_pred.reshape(-1,))