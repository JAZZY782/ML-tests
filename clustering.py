import pandas as pd
import numpy as np
df=pd.read_csv("Social_Network_Ads.csv")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df['EstimatedSalary']=sc.fit_transform(df[['EstimatedSalary']])
df['Age']=sc.fit_transform(df[['Age']])
from sklearn.cluster import KMeans
km=KMeans()
km.fit(df[['EstimatedSalary','Age']])
y_predicted=km.predict(df[['EstimatedSalary','Age']])
print(y_predicted)
df['cluster']=y_predicted
print(df)
