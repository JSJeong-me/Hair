import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

# 1. Synthetic 3D data 생성 (3개 클래스)
np.random.seed(42)
n_samples = 100
class1 = np.random.multivariate_normal([0.3, 0.2, 0.1], 0.05*np.eye(3), n_samples)
class2 = np.random.multivariate_normal([-0.3, -0.2, 0.1], 0.05*np.eye(3), n_samples)
class3 = np.random.multivariate_normal([0.0, 0.3, -0.3], 0.05*np.eye(3), n_samples)

X = np.vstack((class1, class2, class3))
y = np.array([0]*n_samples + [1]*n_samples + [2]*n_samples)

# 2. PCA (3D → 3D for visualization)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# 3. LDA (3D → 2D)
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_pca, y)

# 4. SVM (linear kernel on LDA transformed data)
svm = SVC(kernel='linear')
svm.fit(X_lda, y)

# 5. Visualization
fig = plt.figure(figsize=(18, 5))

# ---- Plot 1: PCA result (3D scatter)
ax1 = fig.add_subplot(131, projection='3d')
scatter1 = ax1.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=y, cmap='rainbow', s=20)
ax1.set_title("PCA (3D projection)")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_zlabel("PC3")

# ---- Plot 2: LDA result (2D scatter)
ax2 = fig.add_subplot(132)
scatter2 = ax2.scatter(X_lda[:,0], X_lda[:,1], c=y, cmap='rainbow', s=20)
ax2.set_title("LDA (2D projection)")
ax2.set_xlabel("LD1")
ax2.set_ylabel("LD2")

# ---- Plot 3: SVM Decision Boundaries on LDA space
ax3 = fig.add_subplot(133)
scatter3 = ax3.scatter(X_lda[:,0], X_lda[:,1], c=y, cmap='rainbow', s=20)

# Create grid to plot decision boundary
xx, yy = np.meshgrid(np.linspace(X_lda[:,0].min()-0.5, X_lda[:,0].max()+0.5, 200),
                     np.linspace(X_lda[:,1].min()-0.5, X_lda[:,1].max()+0.5, 200))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax3.contourf(xx, yy, Z, alpha=0.2, cmap='rainbow')
ax3.set_title("SVM Decision Boundaries (LDA space)")
ax3.set_xlabel("LD1")
ax3.set_ylabel("LD2")

plt.tight_layout()
plt.show()
