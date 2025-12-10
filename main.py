
import matplotlib.pyplot as plt
import numpy as np
from preprocess import load_data
from model import KMeansScratch
from sklearn.cluster import KMeans as SKKMeans

def elbow(X):
    inertias = []
    ks = range(1,11)
    for k in ks:
        km = KMeansScratch(n_clusters=k)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.plot(ks, inertias, marker='o')
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.savefig("elbow.png")
    return inertias

def run():
    X,_ = load_data()
    inertias = elbow(X)
    opt_k = inertias.index(min(inertias[1:], key=lambda x:abs(x-inertias[0])))+1
    if opt_k < 2: opt_k = 3

    km_s = KMeansScratch(n_clusters=opt_k).fit(X)
    km_sk = SKKMeans(n_clusters=opt_k, random_state=42).fit(X)

    np.savetxt("scratch_centroids.csv", km_s.cluster_centers_, delimiter=",")
    np.savetxt("sklearn_centroids.csv", km_sk.cluster_centers_, delimiter=",")

    print("Scratch inertia:", km_s.inertia_)
    print("Sklearn inertia:", km_sk.inertia_)
    print("Optimal k:", opt_k)

if __name__ == "__main__":
    run()
