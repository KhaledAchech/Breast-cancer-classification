import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
"""
this program will help classify breast cancers into malignant or benign
"""


#here am going to import the breast cancer datasets
cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

#preparing the train and test models
x = cancer.data
y = cancer.target


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.33)
#print(x_test,y_train)

#the classification will be based on these two Malignant and Benign
classes = ['malignant' 'benign']

#getting the classification from SVM algorithme
clf = svm.SVC(kernel="linear", C=1)

#fitting the train models to the svm classification
clf.fit(x_train, y_train)
#getting the prediction
y_pred = clf.predict(x_test)

#now we are going to get our accuracy score :)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)

#printing the results :)
for i in range(len(y_pred)):
    print("predicted : ", y_pred[i])
    print("Data : ", x_test[i])
    print("Actual : ", y_test[i])