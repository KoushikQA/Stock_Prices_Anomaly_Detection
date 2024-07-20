import importPackages

prices = pd.read_csv("prices.csv")

prices.head()

prices.fillna(0,inplace=True)

import seaborn as sns
sns.heatmap(prices.corr())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_prices = scaler.fit_transform(prices)

# Applying PCA

# plotting the explained variance ratio.
# Representing the components which preserve maximum information and plot to visualize
# Compute the daily returns of the 500 company stocks.
# Plot the stocks with most negative and least negative PCA weights.
# Use reference as above. Discuss the least and most impacted industrial sectors in terms of stocks,
# during the pandemic period (Year 2020)

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(scaled_prices)

table = [prices.columns.values[1:],pca.components_[0]]
pd.DataFrame(pca.components_.T,columns=['component '+str(i+1) for i in range(394)]
             ,index=prices.columns.values)

plt.bar(range(len(pca.explained_variance_ratio_[:20])), pca.explained_variance_ratio_[:20])
plt.xlabel('components')
plt.show()

pc1 = pd.Series(index=prices.columns, data=pca.components_[0])
pc1.plot(figsize=(10,6), xticks=[], grid=True, title='First Principal Component of the S&P500')

fig, ax = plt.subplots(2,1, figsize=(24,20))
pc1.nsmallest(10).plot.bar(ax=ax[0], color='green', grid=True, title='Stocks with Most Negative PCA Weights')
pc1.nlargest(10).plot.bar(ax=ax[1], color='blue', grid=True, title='Stocks with Least Negative PCA Weights')

# Apply T-SNE [t-distributed Stochastic Neighbor Embedding]
# for dimensionality reduction and data visualization
# and visualize with a graph

from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2, random_state = 0)
tsne_data = tsne.fit_transform(prices)
print(tsne_data.shape)
plt.scatter(tsne_data[:,0],tsne_data[:,1])
plt.show()











