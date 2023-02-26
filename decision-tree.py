import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import graphviz 

df = pd.read_csv('D:/machine learning/PlayTennis.csv')
#print(df.dtypes)

print(df.shape)
Le = LabelEncoder()
df['outlook'] = Le.fit_transform(df['outlook'])
df['temp'] = Le.fit_transform(df['temp'])
df['humidity'] = Le.fit_transform(df['humidity'])
df['windy'] = Le.fit_transform(df['windy'])
df['play'] = Le.fit_transform(df['play'])

print(df.head())
y = df['play']
X = df.drop(['play'], axis= 1)

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, y)
tree.plot_tree(clf)



dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 


X_pred = clf.predict([['2','1','0','0']])

print(X_pred)