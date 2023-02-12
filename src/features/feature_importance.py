from sklearn.inspection import permutation_importance
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,mean_squared_error,make_scorer,r2_score,mean_absolute_error,recall_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer
from keras.callbacks import EarlyStopping
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
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
from pathlib import Path
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


data = pd.read_pickle("../../data/processed/data_processed_onehot_encoded_wo_outliers_train-test.pickle")
y= data["gun_sayisi"]
X =data.drop(["gun_sayisi"],axis=1)
X_df = data.drop(["gun_sayisi"],axis=1)

scaler = pickle.load(open("../../models/neural_network(mae 7.99)_scaler.pkl",'rb'))

#X = data.drop(['is_emri'],axis=1)
X = scaler.transform(X)

model = keras.models.load_model("../../models/neural_network(mae 7.99)")


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.25,random_state=42)
# X is your feature matrix, y is your target vector
# train_test_split and model fitting are not shown for brevity



def keras_score(y_true, y_pred):
    return mean_absolute_error(y_true=y_true, y_pred=y_pred)

# Define the custom scoring function
scoring = make_scorer(keras_score, greater_is_better=False)

# Compute feature importance scores
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42,scoring=scoring)

# Print feature importance scores
sorted_idx = result.importances_mean.argsort()
for i in sorted_idx:
    print(f"{X_df.columns[i]:<20} {result.importances_mean[i]:.3f} +/- {result.importances_std[i]:.3f}")

# Normalize the feature importance scores
importance_normalized = result.importances_mean / np.sum(result.importances_mean)

# Print the normalized feature importance scores
perm_results=[]
cum_imp=0
for i in importance_normalized.argsort()[::-1]:
    cum_imp =importance_normalized[i]+cum_imp
    if cum_imp<=0.99:
        print(f"onemli {X_df.columns[i]:<8} {importance_normalized[i]:.3f}")
        perm_results.append(f"{X_df.columns[i]:<8} {importance_normalized[i]:.3f}")
    elif cum_imp>0.99:
        print(f"onemsiz {X_df.columns[i]:<8} {importance_normalized[i]:.3f}")
        perm_results.append(f"{X_df.columns[i]:<8} {importance_normalized[i]:.3f}")


pd.DataFrame(perm_results).to_excel("../../reports/feature_importance.xlsx",index=False)