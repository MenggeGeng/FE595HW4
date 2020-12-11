#part 2: Both the Wine and the Iris datasets are built with the assumption that there are 3 populations in the sample.
#Using the elbow heuristic, demonstrate with a graph that 3 is the correct (or incorrect) number of populations to use.

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd


#---------load_iris part---------
#load iris dataset
iris = load_iris()
#print(iris)
print(iris['feature_names'])
print(iris['target_names'])

data_iris = pd.DataFrame(iris['data'], columns= iris['feature_names'])
print(data_iris.head())

#build clustering model by using KMeans
#Using inertia_ to validate the performance of K
#inertia_ : Sum of squared distances of samples to their closest cluster center.
inertia = []
for k in range(1,16):
    kmodel = KMeans(n_clusters= k, random_state=2)
    kmodel.fit(data_iris)
    inertia.append(kmodel.inertia_)

#the elbow plot of the inertia
Num_of_clust = range(1,16)
plt.plot(Num_of_clust,inertia)
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('The KMeans clustering of iris')
plt.show()

#The descend rate of inertia:
d_rate = []
len_r = len(inertia)
for i in range(0,len_r-1):
    d_r = 1-(inertia[i+1]/inertia[i])
    d_rate.append(d_r)
print(d_rate)
#After K=3, the descend rate is getting slower.Therefore, 3 is the correct number of populations to use.


#---------load_wine part---------
wine = load_wine()
print(wine['feature_names'])
print(wine['target_names'])

data_wine = pd.DataFrame(wine['data'], columns= wine['feature_names'])
print(data_wine.head())

#build clustering model by using KMeans
inertia_w = []
for k in range(1,16):
    kmodel_w = KMeans(n_clusters= k, random_state=3)
    kmodel_w.fit(data_wine)
    inertia_w.append(kmodel_w.inertia_)

#the elbow plot of the inertia
Num_of_clust = range(1,16)
plt.plot(Num_of_clust,inertia_w)
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('The KMeans clustering of wine')
plt.show()

#The descend rate of inertia:
d_rate_w = []
len_w = len(inertia_w)
for i in range(0,len_w-1):
    d_r = 1-(inertia_w[i+1]/inertia_w[i])
    d_rate_w.append(d_r)
print(d_rate_w)
#After K=3, the descend rate is getting slower.Therefore, 3 is the correct number of populations to use.


