import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# load MNIST dataset
X, y = datasets.load_digits(return_X_y=True)
print('Dimensions before PCA:', X.shape)

# use PCA to reduce dimension from 64 to 2
pca_2d = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=0))
pca_2d.fit(X, y)
X_pca_2d = pca_2d.transform(X)
print('Dimensions after PCA-2D:', X_pca_2d.shape)

# use PCA to reduce dimension from 64 to 3
pca_3d = make_pipeline(StandardScaler(), PCA(n_components=3, random_state=0))
pca_3d.fit(X, y)
X_pca_3d = pca_3d.transform(X)
print('Dimensions after PCA-3D:', X_pca_3d.shape)

# # use tSNE to reduce dimension from 64 to 2
# tsne = make_pipeline(StandardScaler(), TSNE(n_components=2, init='pca', random_state=0))
# tsne.fit(X, y)
# X_tsne_2d = tsne.fit_transform(X)
# print('Dimensions after tSNE-2D:', X_tsne_2d.shape)
#
# # use tSNE to reduce dimension from 64 to 3
# tsne = make_pipeline(StandardScaler(), TSNE(n_components=3, init='pca', random_state=0))
# tsne.fit(X, y)
# X_tsne_3d = tsne.fit_transform(X)
# print('Dimensions after tSNE-3D:', X_tsne_3d.shape)

# plot the points projected with PCA and tSNE
fig = plt.figure(figsize=(18, 8))
fig.suptitle('MNIST Visualization')

ax = fig.add_subplot(121)
ax.title.set_text('PCA-2D')
scatter = ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, s=30, cmap='Set1')
legend = ax.legend(*scatter.legend_elements(),
                    loc="lower right", title="Digit")
ax.add_artist(legend)

ax = fig.add_subplot(122, projection='3d')
ax.title.set_text('PCA-3D')
ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=y, cmap='Set1')
legend = ax.legend(*scatter.legend_elements(),
                    loc="lower right", title="Digit")
ax.add_artist(legend)
# ax = fig.add_subplot(223)
# ax.title.set_text('tSNE-2D')
# ax.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=y, s=30, cmap='Set1')
#
# ax = fig.add_subplot(224, projection='3d')
# ax.title.set_text('tSNE-3D')
# ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=y, cmap='Set1')\

plt.legend(loc='lower right')
plt.savefig('3d_plot.png')
plt.show()