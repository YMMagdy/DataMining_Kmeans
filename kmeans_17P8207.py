
#Importing and reading from the csv file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
df= pd.read_csv('./Mall_Customers.csv',index_col="CustomerID")
X=df.iloc[:,[2,3]].values
#Implementing the Elbow method to know the number of clusters
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(10,5))
sns.lineplot(range(1,11),wcss,marker="o",color='red')
plt.title('Elbow Method')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()

#Number of clusters known from the previous plot using the elbow method
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
#Training
y_pred=kmeans.fit_predict(X)
plt.figure(figsize=(10,6))
for i in range(5):
    plt.scatter(X[y_pred==i,0],X[y_pred==i,1],label='cluster'+str(i+1))
    plt.legend()
    
#Plotting the results of the clustering
plt.grid(False)
plt.title('Clusters of Customers')
plt.xlabel('Annual Income in k$')
plt.ylabel('Spending')
plt.show()