from lib import *

df_train_data = pd.read_csv('D:/My Projects/Predict house price/train.csv')
df_test_data  = pd.read_csv('D:/My Projects/Predict house price/test.csv')


#print(df_train_data.head())
 
category = df_train_data.loc[:, df_train_data.columns != 'SalePrice']
category = category.select_dtypes(include='object')
category = category.fillna('None')

enc = OneHotEncoder(handle_unknown = 'ignore').fit(category)
category = enc.transform(category).toarray()
category_sub  = enc.transform(df_test_data.select_dtypes(include=['object']).fillna('None')).toarray()


X = df_train_data.loc[:, df_train_data.columns != "SalePrice"]
X = X.select_dtypes(include=['float64', 'float', 'int']).fillna(df_train_data.mean())# return to mean value in the required axis
X = X.to_numpy()
X_sub = df_test_data.select_dtypes(include=['float64', 'float', 'int']).fillna(df_test_data.mean()).to_numpy()
y = df_train_data["SalePrice"].to_numpy()


# scale numerical  value
sc = StandardScaler()
X = sc.fit_transform(X)
X_sub = sc.fit_transform(X_sub)

#concate category and numerical value
X = np.concatenate((X, category), axis = 1)
X_sub = np.concatenate((X_sub, category_sub), axis = 1)




# split train and test
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# create model
number_of_input = X_train.shape[1]
input_tensor = Input(shape = (number_of_input))
x = Dense(512, activation = 'relu')(input_tensor)
x = Dense(512, activation = 'relu')(x)
x  = Dense(512, activation = 'relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(1)(x)
model = Model(inputs = input_tensor, outputs = output_tensor)
model.compile(optimizer = 'adam', loss = 'mse')

model.summary()

# train model
model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size  = 128, epochs = 1)



# visualization

losses = pd.DataFrame(model.history.history)
plt.figure(figsize = (15,5))
sns.lineplot(data = losses, lw = 3)
plt.xlabel('Epochs')
plt.title('Trainign loss per Poch')
sns.despine()
plt.show()

# Predict

prediction = model.predict(X_sub)
ids = df_test_data['Id'].to_numpy()
ids = ids.reshape((1459,1))
sub  = np.concatenate((ids, prediction),axis = 1)
sub_df = pd.DataFrame(sub, columns = ['ID','SalePrice'])
sub_df.to_csv('D:/My Projects/Predict house price/submission.csv', index = False)
