from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import accuracy_score

X_train, y_train = datasets.load_svmlight_file("a9a.txt")
X_test, y_test = datasets.load_svmlight_file("a9a.t")
svm_c = svm.SVC(C = 1, gamma = 0.1, kernel = "rbf")

svm_c.fit(X_train, y_train)

# Step 6: Predict
y_pred = svm_c.predict(X_test)

# Step 7: Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("Cross-validation score (avg):", accuracy)