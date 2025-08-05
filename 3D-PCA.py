# 전체 코드: 데이터 생성 → PCA → LDA → SVM → 3D 시각화 (2개 subplot)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

# 1. Synthetic dataset 생성 (3 classes)
np.random.seed(0)
n_samples = 100
class1 = np.random.multivariate_normal([0.5, 0.5, 0.1], 0.08*np.eye(3), n_samples)
class2 = np.random.multivariate_normal([-0.3, -0.2, 0.2], 0.08*np.eye(3), n_samples)
class3 = np.random.multivariate_normal([0.0, 0.3, -0.3], 0.08*np.eye(3), n_samples)

X = np.vstack((class1, class2, class3))
y = np.array([0]*n_samples + [1]*n_samples + [2]*n_samples)

# 2. PCA 적용 (3D)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# 3. LDA 적용 (2D → 시각화 전처리용)
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_pca, y)

# 4. SVM 학습 (PCA 3D 공간에서)
svm_3d = SVC(kernel='linear')
svm_3d.fit(X_pca, y)

# 5. SVM 초평면 계산
w = svm_3d.coef_[0]
b = svm_3d.intercept_[0]
xx, yy = np.meshgrid(np.linspace(X_pca[:,0].min()-0.5, X_pca[:,0].max()+0.5, 30),
                     np.linspace(X_pca[:,1].min()-0.5, X_pca[:,1].max()+0.5, 30))
zz = (-w[0]*xx - w[1]*yy - b)/w[2]

# 6. 시각화 (2개 subplot)
fig = plt.figure(figsize=(16,6))

# Subplot 1: PCA 3D 데이터 분포
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=y, cmap='rainbow', s=25)
ax1.set_title("PCA 3D Projection (Data Distribution)")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_zlabel("PC3")

# Subplot 2: PCA 3D + SVM 초평면
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=y, cmap='rainbow', s=25)
ax2.plot_surface(xx, yy, zz, alpha=0.3, color='gray')
ax2.set_title("PCA 3D Projection + SVM Hyperplane")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_zlabel("PC3")

plt.tight_layout()
plt.show()
