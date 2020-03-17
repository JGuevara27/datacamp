from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
iris = datasets.load_iris()

(type(iris)) # shows type of iris which is a bunch, i.e. key - value pairs
(iris.keys())
type(iris.data) # numpy.ndarray
type(iris.target) # numpy.ndarray
iris.data.shape # (150,4) > 150 rows, and 4 columns
# samples are in rows, features are in columns

iris.target_names # gives you array with types of irises

# Exploratory data analysis (EDA)
X = iris.data
Y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
# print(df.head())

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])

X_new = np.array([[5.6, 2.8, 3.9, 1.1], [5.7, 2.6, 3.8, 1.3], [4.7, 3.2, 1.3, 0.2]])
X_new.shape # >>> 3 rows and 4 columns
prediction = knn.predict(X_new)
knprint(prediction) # result is 1, 1, 0 which are the types of plants setosa etc.K
