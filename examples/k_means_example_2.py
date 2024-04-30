from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from mlib.algorithms.k_means import KMeans

def k_means_example_2():
    X, _ = make_blobs(
        n_samples=200,
        n_features=2,
        centers=3,
        cluster_std=0.2,
        shuffle=True
    )
    
    # Elbow method
    inertias = []

    for i in range(1, 20):
        k_means = KMeans(k=i)
        k_means.fit(X)
        inertias.append(k_means.inertia)

    plt.plot(range(1, 20), inertias)
    plt.title('Elbow method')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.show()

    # Display clusters
    k_means = KMeans(k=3)
    k_means.fit(X)
    labels = k_means.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()

k_means_example_2()