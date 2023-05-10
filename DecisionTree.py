import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Importing the dataset
dataset = pd.read_csv('flags.csv')
X = dataset.iloc[:, 7:26].values
y = dataset.iloc[:, 29].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# fitting decision tree classifier to the training set
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# predict test set result
y_pred = classifier.predict(X_test)
print('Số lượng mẫu dự đoán chính xác: ', accuracy_score(y_test, y_pred, normalize=False))
print('Số lượng mẫu kiểm thử : ', len(y_test))
print('Độ chính xác: ', round(accuracy_score(y_test, y_pred)*100, 2), '%')


