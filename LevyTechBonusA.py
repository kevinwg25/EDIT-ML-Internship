# Kevin Wang Levy's Lab Bonus Quest A
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('')
x = data.iloc[:, 1:].values
y = data.iloc[:, 0].values
#split data into features x and y for rows and columns respectivity

pca = PCA(n_components=2)
xpca = pca.fit_transform(x)
#PCA object to fit to features x
tsne = TSNE(n_componenets=2)
xtsne = tsne.fit_transform(x)
#tSNE object to fit to features x, transforms x into reduced dimensional space

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(xpca[:, 0], xpca[:, 1], c=pd.factorize(y)[0], cmap='viridis')
plt.title('PCA')
plt.xlabel('Principal Component 1')
plt.xlabel('Principal Component 2')

plt.subplot(1, 2, 2)
plt.scatter(xtsne[:, 0], xtsne[:, 1], c=pd.factorize(y)[0], cmap='viridis')
plt.title('t-SNE')
plt.xlabel('Principal Component 1')
plt.xlabel('Principal Component 2')

plt.tight_layout()
plt.show()

""" training """

xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.2, random_state=42)
#split intro training and validation sets. Allocates 80% training 20% validation

#svmmodel = SVC(kernel='rbf', gamma='scale')
#svmmodel.fit(xtrain, ytrain)
#Support vector classifier with kernal fit to the training data

kmeans = KMeans(n_cluster=2, random_state=2)
kmeans.fit(xtrain)
ypred = kmeans.predict(xval)

accuracy = accuracy_score(yval, ypred)
print("Accuracy: ", accuracy)
classification = classification_report(yval, ypred)
print("Classification Report: ")



