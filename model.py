import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle


dataset = pd.read_csv('heart.csv')

y = dataset['target']
X = dataset.drop(['target'], axis=1)  # removing the target field for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)  # spliting the data for test and train
# Feature Scaling  
from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test) 

#knn (k-nearest neighbors)
knn = KNeighborsClassifier(n_neighbors = 9)
knn_model=knn.fit(X_train, y_train)
knn_predict = knn_model.predict(X_test)
print(knn_predict)
knn_score = round(accuracy_score(knn_predict,y_test)*100,2)
print("The accuracy score achieved using KNN is: "+str(knn_score)+" %")
print("Model is trained and ready to use")
score = knn_model.score(X_test, y_test)
print(score)
# # Make pickle file of our model
pickle.dump(knn, open("model.pkl", "wb"))
