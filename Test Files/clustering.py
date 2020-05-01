# load the modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import kneighbors_graph
#from sklearn.cluster import KMeans, MeanShift
import prims
from cvxpy import *

# distance metric
def alpha_metric(Xi, Xj):
    global alpha
    return np.sqrt((Xi[0]-Xj[0])**2 + (Xi[1] - Xj[1])**2) + alpha*np.abs(Xi[2]-Xj[2])

# adjacency matrix
def adjacency(X, alphas):
    G = []
    for a in alphas:
        global alpha
        alpha = a
        G.append(kneighbors_graph(X, n_neighbors=100, mode='distance', metric='pyfunc', metric_params={'func':alpha_metric}))
    return G

# method to plot a MST with specified edges cut
def plot_mst(A, one, two, a):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(A[:,0], A[:,1], alpha=0.1)
    plt.quiver(A[one, 0], A[one, 1], A[two, 0] - A[one, 0], A[two, 1] - A[one, 1],
                angles='xy', scale_units='xy', scale=1, headwidth=0, headaxislength=0, headlength=0, minlength=0)
    ax.text(0.05, 0.95, 'alpha: %d%%' % a, transform=ax.transAxes, va='top')
    fig.tight_layout()

# fitting data for alpha vs distance to: dist = -A*(alphas-x0)**2 + y0
def fit(alphas, A, x0, y0):
    return -A*(alphas - x0)**2 + y0

# main method
def main():
    # load the data
    data = np.load("clusters.npy")
    ra0, dec0 = data['RA'].mean(), data['DEC'].mean()
    X = np.dstack(((ra0-data['RA'])*np.cos(np.radians(data['DEC'])), data['DEC']-dec0, data['Z']))[0]  
    m = len(data)   # number of data points
    print(m, data.dtype.names)

    # creating adjacency matrices for different values of alpha
    alphas = np.linspace(0, 1, 50)
    G = adjacency(X, alphas)

    cost = []
    for i in range(len(G)):
        a = alphas[i]
        A = G[i]
        print("\nAlpha: " + str(a))

        # determine the edges to cut
        lengths = []
        for i in range(m):
            for j in range(i, m):
                l = A[i, j]
                if (l != 0):
                    lengths.append(l)
        mean = np.mean(lengths)
        mad = stats.median_absolute_deviation(lengths)
        threshold = -1.2

        # run Prim's algorithm to find clusters
        mst, one, two, K, clusters, distances = prims.cluster(A, m, mean, mad, threshold)
        print("Number of clusters: " + str(K))

        # mean sum of the mean intracluster distances
        dist = []
        for ds in distances:
            for d in distances[ds]:
                dist.append(d)

        # cost is the mean of these distances, normalised for the graph
        print("Alpha: " + str(a))
        print("Mean: " + str(mean))
        print("Cost: " + str(np.mean(dist)/mean))
        cost.append(np.mean(dist)/mean)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    print(alphas, cost)
    ax1.plot(alphas, cost)
    ax1.set_title("How Distance Changes with Alpha")
    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("Distance")
    fig1.tight_layout()
    plt.show()

main()
