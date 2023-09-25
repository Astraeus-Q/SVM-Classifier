from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = datasets.load_svmlight_file("a9a.txt")


para_range = [0.01, 0.05, 0.1, 0.5, 1]

for para_c in para_range:
    for para_g in para_range:
        svm_c = svm.SVC(C = para_c, kernel = "rbf", gamma = para_g)
        score_c = np.mean(cross_val_score(svm_c, X, y, cv=3))
        print(f"C = {para_c}, gamma = {para_g}, Score: {score_c}")
