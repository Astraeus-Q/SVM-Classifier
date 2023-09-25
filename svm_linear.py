from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = datasets.load_svmlight_file("a9a.txt")
svm_c = svm.SVC(C = 0.5, kernel = "linear")


para_range = {0.01, 0.05, 0.1, 0.5, 1}

score_c = np.mean(cross_val_score(svm_c, X, y, cv=3)) # 3-fold

print("Cross-validation score (avg):", score_c)