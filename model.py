import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
warnings.filterwarnings("ignore")

# Reading the dataset
df=pd.read_csv('trainable_complete2.csv')


#Feature Engineering
# Label Encoding 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_mode = df.select_dtypes(include ='object').columns

# For  Dataset
for i in var_mode:
    df[i] = le.fit_transform(df[i])

# Divide dataset into dependent and independent features

# independent features
x=df.drop('optimal_hours',axis=1)

# dependent feature
y=df['optimal_hours']

# Apply feature scailing to the independent features:
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

# Apply principle component analysis
from sklearn.decomposition import PCA
pca=PCA(0.95) # components store 95% of the information
x_pca=pca.fit_transform(x_scaled)

# print(X,y)
x_train,x_test,y_train,y_test=train_test_split(x_pca,y,test_size=0.33,random_state=101)

# Random Forest Regressor
rf=RandomForestRegressor()
rf=rf.fit(x_train,y_train)


pickle.dump(rf,open('model.pkl','wb'))

