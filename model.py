
import numpy as np

class KMeansScratch:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        np.random.seed(random_state)
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids(self, X):
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[idx].copy()

    def _assign(self, X, centers):
        d = np.linalg.norm(X[:,None,:] - centers[None,:,:], axis=2)**2
        labels = np.argmin(d, axis=1)
        return labels, d

    def _update(self, X, labels):
        new = []
        for k in range(self.n_clusters):
            pts = X[labels==k]
            if len(pts)==0:
                new.append(X[np.random.choice(len(X))])
            else:
                new.append(pts.mean(axis=0))
        return np.array(new)

    def fit(self, X):
        centers = self._init_centroids(X)
        for _ in range(self.max_iter):
            labels, dist = self._assign(X, centers)
            new_centers = self._update(X, labels)
            if np.linalg.norm(new_centers - centers) <= self.tol:
                break
            centers = new_centers
        labels, dist = self._assign(X, centers)
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = np.sum(np.min(dist, axis=1))
        return self
