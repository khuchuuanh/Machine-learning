import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score
import category_encoders as ce


df = pd.read_csv('D:/machine learning/weatherAUS.csv')

# check for cardinality in categorical variables

#for var in categorical:
#    print(var, 'contains', len(df[var].unique()), 'labels')


df['Date']  = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

#print(df.info())
df.drop('Date', axis = 1, inplace= True)


categorical = [var for var in df.columns if df[var].dtype == 'O']
print('there are {} categorical variables '.format(len(categorical)))
print('\nthe categorical variables are :', categorical)

missing = df[categorical].isnull().sum().sort_values()
#print(missing)
'''
print('Location cotains', len(df.Location.unique()),'label')
print(df.Location.unique())
print(df.Location.value_counts())

pd.get_dummies(df.Location, drop_first= True)
pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True)
pd.get_dummies(df.WindDir9am, drop_first= True, dummy_na= True)
pd.get_dummies(df.WindDir3pm, drop_first= True, dummy_na= True)
pd.get_dummies(df.RainToday, drop_first= True, dummy_na=True)
# find numerical variabless'''



numerical = [var for var in df.columns if df[var].dtype != 'O']
print('There are {} numerical variable \n'.format(len(numerical)))
print('numerical variables are : ', numerical)

#print(df[numerical].head())
missing_val = df[numerical].isnull().sum().sort_values()
print(missing_val)

#print(round(df[numerical].describe()))

plt.figure(figsize= (15,10))

plt.subplot(2, 2, 1)
fig = df.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = df.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')

plt.subplot(2,2,3)
fig = df.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')

plt.subplot(2, 2, 4)
fig = df.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')




plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = df.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = df.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = df.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = df.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')




#plt.show()
IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)
Lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)
#print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)
Lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)
##print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)
Lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)
#print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)
Lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)
#print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))





X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']

y.fillna(y.mode()[0], inplace= True)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)

#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

#print(X_train.dtypes)

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
#print(categorical)

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']
#print(numerical)

missing_train = X_train[numerical].isnull().sum().sort_values()

missing_test = X_test[numerical].isnull().sum().sort_values()
#print(missing_test)

for df1 in [X_train, X_test]:
    for col in numerical:
        col_median = X_train[col].median()
        df1[col].fillna(col_median, inplace = True)

missing_train = X_train[numerical].isnull().sum().sort_values()
#print(missing_train)
missing_test = X_test[numerical].isnull().sum().sort_values()
#print(missing_test)


for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)

def max_value(df3, variable, top):
    return np.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)




print(X_train.Rainfall.max(),  X_test.Rainfall.max())
print(X_train.Evaporation.max(), X_test.Evaporation.max())
print(X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max())
print(X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max())

#print(X_train[categorical])

encoder = ce.BinaryEncoder(cols  = ['RainToday'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.fit_transform(X_test)



X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location), 
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)

#print(X_train.head())

X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location), 
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)



cols = X_train.columns
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


X_train = pd.DataFrame(X_train,columns= cols)
X_test = pd.DataFrame(X_test, columns= cols)

logreg = LogisticRegression(solver= 'liblinear', random_state= 0)

logreg.fit(X_train, y_train)

y_pred_test = logreg.predict(X_test)
print(y_pred_test)

# predict_proba method
# probability of getting output as 0 - no rain
print(logreg.predict_proba(X_test)[:,0])

# probability of getting output as 1 -rain
print(logreg.predict_proba(X_test)[:,1])

print('The accuracy of model is :{0:0.4f}'.format(accuracy_score(y_test, y_pred_test)))

y_pred_train = logreg.predict(X_train)

print('the accuracy of model is:{0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))



