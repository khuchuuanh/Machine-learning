import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle


df = pd.read_csv('diabetes.csv')


# data processing
#print(df.head())
print(df.info()) # check null value

# check number of classes of target
number_class = df['Outcome']
print(number_class.value_counts())


# check dtype of data
#print(df.dtypes)

# summary data
#print(df.describe())


# data visualization

# histogram for all data
#df.hist() 

# Density plot for all data

#df.plot(kind = 'density', subplots = True, layout =(3,3), sharex = False)


# box plot for all data
#df.plot(kind = 'box', subplots = True, layout =(3,3), sharex = False)


# correlation matrix

#sns.heatmap(data=df.corr(), annot= True)


# scatter matrix plot
#scatter_matrix(df)
#plt.show()


# Data split

target = 'Outcome'
y = df[target]
x = df.drop(target, axis= 1)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state= 42)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

pram_grid = {
    'C' : [1,2,3],
    'kernel' : ['linear', 'poly', 'rbf','sigmoid', 'precomputed'],
    'degree' : [2,3],
    'gamma' : ['scale', 'auto'],
    'coef0' : [0.5,1.0],
    'shrinking' : ['True','False'],
    'probability' : ['True','False'],
    'tol' : [1e-3, 1e-4],
    'max_iter' : [100, -1],
    'decision_function_shape' : ['ovo','ovr'],
    'break_ties' : ['True', 'False']
}

cls = GridSearchCV(SVC(random_state= 42), param_grid= pram_grid )

cls = SVC()
cls.fit(x_train, y_train)
# save model using pickle
#pickle.dump(cls, open('svc_model.pkl', 'wb')) # wr: write binary

# load model
#model = pickle.load(open('svc_model.pkl', 'rb'))
#y_predict = model.predict(x_test)

y_predict = cls.predict(x_test)


# for i, j in  zip(y_test, y_predict):
#     print('y_true',i, 'y_pred', j)


print(cls.class_weight_)
print(cls.classes_)

print(cls.n_iter_)
# print(cls.support_)
# print(cls.support_vectors_)
# classifier report
print('----- classifier report -------------')
print(classification_report(y_test, y_predict))

# confusion matrix
print('----- confusion matrix -------------')
print(confusion_matrix(y_test, y_predict))
