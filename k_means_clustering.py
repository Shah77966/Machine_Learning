import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_excel('C:\\Users\\shahk\\Documents\\Shah\\Programming\\Machine_Learning\\python_for_microscopists-master\\other_files\\K_Means.xlsx')
# print(df.head())

# sns.regplot(x=df['X'], y=df['Y'], fit_reg = False, color='red')
# plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init = "k-means++",max_iter = 300, n_init = 10, random_state=10)

model = kmeans.fit(df)
print(df.head())
predicted_values = kmeans.predict(df)

plt.scatter(df['X'], df['Y'], c=predicted_values, s=50, cmap='viridis')
# plt.show()
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=200, c='black', alpha = 0.5)
# print(kmeans.cluster_centers_[:,0])
# print(kmeans.cluster_centers_[:,1])
plt.show()