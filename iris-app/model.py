from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier()


classifier.fit(x_train,y_train)


predictions=classifier.predict(x_test)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))


import pickle

filename = './models/classifier.sav'
pickle.dump(classifier, open(filename, 'wb'))
