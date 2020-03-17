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

# Visual EDA
_ = pd.plotting.scatter_matrix(df, c = Y, figsize = [8,8], s=150, marker = 'D')
