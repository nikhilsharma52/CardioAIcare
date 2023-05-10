import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


dataset = pd.read_csv('heart.csv')

y = dataset['target']
X = dataset.drop(['target'], axis=1)  # removing the target field for training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)  # spliting the data for test and train
# print("X_train = ", X_train.shape)
# print("X_test = ", X_test.shape)
# print("y_train = ", y_train.shape)
# print("y_test = ", y_test.shape)
# print("Training features have {0} records and Testing features have {1} records.".
#       format(X_train.shape[0], X_test.shape[0]))

rfc = RandomForestClassifier(
    n_estimators=500, criterion='entropy', max_depth=8, min_samples_split=5)
rfc_model = rfc.fit(X_train, y_train)
rfc_predict = rfc_model.predict(X_test)
print(rfc_predict)

# input_data = (54,1,0,122,286,0,0,116,1,3.2,1,2,2)
# 52, 1, 0, 125, 212, 0, 1, 168, 0, 1, 2, 2, 3 --> 0
# 58,	0,	0,	100,	248,	0,	0,	122,	0,	1,	1,	0,	2 --> 1
# 54,1,0,122,286,0,0,116,1,3.2,1,2,2  -->0
#change the input data to a numpy array
# input_data_as_numpy_array = np.asarray(input_data)
# print(input_data_as_numpy_array)
#reshape the numpy array as we are predicting for only one instance
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# prediction = rfc.predict(input_data_reshaped)
# print(prediction)
# if(prediction[0]==0):
#   print('The person doesnt have heart disease')
# else:
#   print('The person has heart disease')

print("Model is trained and ready to use")
# # Make pickle file of our model
pickle.dump(rfc, open("model.pkl", "wb"))
