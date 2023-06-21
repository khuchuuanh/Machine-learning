import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from lazypredict.Supervised import LazyClassifier

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

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])


preprocess = ColumnTransformer(transformers=[
    ('num_feature', num_transformer, ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"])
])

cls = Pipeline(steps=[
    ("preprocess", preprocess), 
    ('model', SVC())
])

param_grid = {
    'model__C': [1, 2, 3],
    'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    "preprocess__num_feature__imputer__strategy": ['mean', 'median']
}

cls_cv = GridSearchCV(cls, param_grid, verbose= 1, n_jobs=4, scoring="accuracy",cv = 6)
#cls = SVC()
cls_cv.fit(x_train, y_train)
# save model using pickle
#pickle.dump(cls, open('svc_model.pkl', 'wb')) # wr: write binary

# load model
#model = pickle.load(open('svc_model.pkl', 'rb'))
#y_predict = model.predict(x_test)

y_predict = cls_cv.predict(x_test)


print(cls_cv.best_estimator_)
print(cls_cv.best_score_)
print(cls_cv.best_params_)


for i, j in  zip(y_test, y_predict):
    print('y_true',i, 'y_pred', j)



# print(cls.support_)
# print(cls.support_vectors_)
# classifier report
print('----- classifier report -------------')
print(classification_report(y_test, y_predict))

# confusion matrix
print('----- confusion matrix -------------')
print(confusion_matrix(y_test, y_predict))
