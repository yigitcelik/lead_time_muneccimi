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

data =pd.read_pickle("../../data/processed/data_processed_onehot_encoded_wo_outliers_train-test.pickle")
data_val =pd.read_pickle("../../data/processed/data_processed_onehot_encoded_wo_outliers_validation.pickle")
# Load the data

y = data["gun_sayisi"]
X = data.drop(["gun_sayisi"],axis=1)

y_val = data_val["gun_sayisi"]
X_val = data_val.drop(["gun_sayisi"],axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

pickle.dump(scaler, open('../../models/neural_network(mae 9.1)_scaler.pkl', 'wb'))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

#%%
with tf.device('/cpu:0'):

    # Build the model
    model = keras.Sequential([
        keras.layers.Dense(units=32, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=1)
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),\
    loss='mean_absolute_error')

    # Define the early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # Train the model
    epochs = 200
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64,
    validation_data=(X_val, y_val),verbose=1,callbacks=[early_stopping])


# Evaluate the model
score = model.evaluate(X_test, y_test, batch_size=64)
# save model

model.save('../../models/neural_network(mae 9.1)')
model = keras.models.load_model('../../models/neural_network(mae 9.1)')

pickle.dump(model, open('../../models/neural_network(mae 9.1).pkl', 'wb'))

pickled_model = pickle.load(open('../../models/neural_network(mae 9.1).pkl', 'rb'))

y_pred_val = pickled_model.predict(X_val).reshape(-1,)
y_pred_test = pickled_model.predict(X_test).reshape(-1,)

def reg_scatter(y_pred,y_test):
    acc_ind = np.abs(y_pred-y_test)<=10

    plot_df =pd.DataFrame({'Gercek lead time':y_test,'Tahmin Lead time':y_pred,'Fark 10 gün icinde':acc_ind})

    fig3 = px.scatter(data_frame=plot_df,x="Gercek lead time",y="Tahmin Lead time",color="Fark 10 gün icinde")

    category_count = plot_df.groupby('Fark 10 gün icinde').size().reset_index(name='count')

    title = category_count['Fark 10 gün icinde'].astype(str) + " = " + category_count['count'].astype(str)
    title = ", ".join(title)

    fig3.update_layout(
        title=title,
    )
    return fig3


reg_scatter(y_pred_val,y_val).write_html("../../reports/figures/neural_network_val_data.html")
reg_scatter(y_pred_test,y_test).write_html("../../reports/figures/neural_network_traintest_data.html")

def class_accuracy(y_pred,y_test):
    acc_ind = np.abs(y_pred-y_test)<=10
    plot_df =pd.DataFrame({'Gercek lead time':y_test,'Tahmin Lead time':y_pred,'Fark 10 gün icinde':acc_ind})
    plot_df["Gercek sınıf"]= pd.cut(plot_df["Gercek lead time"],bins=[0,15,30,2000],labels=["erken","orta","gec"])
    plot_df["Tahmin sınıf"]= pd.cut(plot_df["Tahmin Lead time"],[0,15,30,2000],labels=["erken","orta","gec"])
    return plot_df.dropna()


plot_df = class_accuracy(y_pred_val,y_val)
report = classification_report(y_true=plot_df["Gercek sınıf"].astype('str'),y_pred=plot_df["Tahmin sınıf"].astype('str'),output_dict=True)
report_df = pd.DataFrame(report).transpose()
cm =confusion_matrix(y_true=plot_df["Gercek sınıf"].astype('str'),y_pred=plot_df["Tahmin sınıf"].astype('str'),labels=["erken","orta","gec"])

sns.heatmap(report_df.iloc[:, :-1], annot=True)
plt.savefig("../../reports/figures/neural_network_classreport0-15-30.png")

# %%
