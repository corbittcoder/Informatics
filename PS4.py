import pandas as pd
import numpy as np

#1.1.1load data
df = pd.read_csv('wdbc.csv.bz2').drop(['id'], axis=1)
print(df.columns)

#1.1.2. Table with min and max for each feature
print(df.describe())

#1.1.3. Accuracy of majority classifier
print((df['diagnosis'] == "B").mean())
#The accuracy would be .627 since 62.7% of the data is in the majority class.

#1.2 Feature Transformation
#1.2.1 using texture.mean and concpoints.mean
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
X = df[['texture.mean', 'concpoints.mean']]
y = df['diagnosis']
m = knn.fit(X, y)

#1.2.1 b decision boundary
import matplotlib.pyplot as plt
x1 = df["texture.mean"]
x2 = df["concpoints.mean"]
y = df.diagnosis == "M"
X = np.column_stack((x1, x2))
m = knn.fit(X, y)
def graph(x1, x2, y, m):
    # Now create the regular grid
    ex1 = np.linspace(x1.min(), x1.max(), 10) # 30 elements
    ex2 = np.linspace(x2.min(), x2.max(), 10)
    xx1, xx2 = np.meshgrid(ex1, ex2)
    # meshgrid creates two 30x30 matrices
    g = np.column_stack((xx1.ravel(), xx2.ravel()))
    # we create the design matrix by stacking the xx1, xx2
    # after unraveling those into columns
    # predict on the grid
    hatY = m.predict(g).reshape(10, 10)
    # imshow wants a matrix, so we reshape the predicted vector into one
    _ = plt.imshow(hatY, extent=(x1.min(), x1.max(), x2.min(), x2.max()),
        aspect="auto", interpolation='none', origin='lower',
        # you need to specify that the image begins from below, not above, otherwise it will be flipped around
        alpha=0.3)
    _ = plt.scatter(x1, x2, c=y, edgecolor='k', s=8)
    _ = plt.show()
#graph(x1, x2, y, m)

#the x axis covers a much wider range than the y axis, so the decision boundary is vertical since the y axis barely matters.
#1.2.1 c cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores.mean())

#1.2.2 normalize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores.mean())
m = knn.fit(X, y)
#graph(X[:, 0], X[:, 1], y, m)

#1.2.3 Why does the plot look different? Because the features have been scaled so the y-axis can make an impact on accuracy now. That's why the accuracy is higher in this model.

#1.2.4 Repeat with mahalanobis distance
x1 = df["texture.mean"]
x2 = df["concpoints.mean"]
X = np.column_stack((x1, x2))
Sigma = np.cov(X, rowvar=False)
knn = KNeighborsClassifier(n_neighbors=7, metric='mahalanobis', metric_params={'V':Sigma})
m = knn.fit(X, y)
#graph(df["texture.mean"], df["concpoints.mean"], y, m)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores.mean())

#1.2.5 Repeat with different features
df['diagnosis'] = df.diagnosis == "M"
x1 = df["radius.mean"]
x2 = df["perimeter.mean"]
X = np.column_stack((x1, x2))
y = df['diagnosis']
knn = KNeighborsClassifier()
m = knn.fit(X, y)
#graph(df["radius.mean"], df["perimeter.mean"], y, m)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores.mean())

X = scaler.fit_transform(X)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores.mean())
m = knn.fit(X, y)
# graph(X[:, 0], X[:, 1], y, m)

x1 = df["radius.mean"]
x2 = df["perimeter.mean"]
X = np.column_stack((x1, x2))
Sigma = np.cov(X, rowvar=False)
knn = KNeighborsClassifier(n_neighbors=7, metric='mahalanobis', metric_params={'V':Sigma})
m = knn.fit(X, y)
#graph(x1, x2, y, m)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores.mean())

#symmetry and compactness
x1 = df["symmetry.mean"]
x2 = df["compactness.mean"]
X = np.column_stack((x1, x2))
y = df['diagnosis']
knn = KNeighborsClassifier()
m = knn.fit(X, y)
graph(x1, x2, y, m)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores.mean())

X = scaler.fit_transform(X)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores.mean())
m = knn.fit(X, y)
graph(X[:, 0], X[:, 1], y, m)

x1 = df["symmetry.mean"]
x2 = df["compactness.mean"]
X = np.column_stack((x1, x2))
Sigma = np.cov(X, rowvar=False)
knn = KNeighborsClassifier(n_neighbors=7, metric='mahalanobis', metric_params={'V':Sigma})
m = knn.fit(X, y)
graph(x1, x2, y, m)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores.mean())

#1.2.6 explanation
# Euclidean distance does better in general when the features are scaled, although we did see one exception, probably where the disproportionate feature was a good predictor of the outcome.
# Mahalanobis distance always did the best though, which makes sense since many of these features are correlated.
# The graphs matched the accuracy well, although the Mahalanobis did create weird isolated squares sometimes.