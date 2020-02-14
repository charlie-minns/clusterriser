# load the modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import kneighbors_graph
#from sklearn.cluster import KMeans, MeanShift
from cvxpy import *

# distance metric
def alpha_metric(Xi, Xj):
    global alpha
    return np.sqrt((Xi[0]-Xj[0])**2 + (Xi[1] - Xj[1])**2) + alpha*np.abs(Xi[2]-Xj[2])

# method to plot a MST with specified edges cut
def plot_mst(one, two, p):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(X[:,0], X[:,1], alpha=0.1)
    plt.quiver(X[one, 0], X[one, 1], X[two, 0] - X[one, 0], X[two, 1] - X[one, 1],
                angles='xy', scale_units='xy', scale=1, headwidth=0, headaxislength=0, headlength=0, minlength=0)
    ax.text(0.05, 0.95, 'q: %d%%' % p, transform=ax.transAxes, va='top')
    fig.tight_layout()

# an edge object for the edges in a graph
class Edge():
    '''An edge in a graph'''
    def __init__(self, i, j, weight):
        self.vFrom = i
        self.vTo = j
        self.weight = weight
        self.visited = False 

# running Prim's algorithm to find the mst in a graph G with V vertices
def prims(A, m, mean, mad, threshold):
    mst = []                            # list of edges in the mst
    one = []
    two = []
    edges = []                          # list of edges from visited vertices
    v = 0                               # starting algorithm from a random vertex
    visited = []                        # keeping track of the vertices that have been visited
    distances = []
    K = 1
    clusters = {}
    clusters[K] = []

    # go through vertices until an mst is found
    while(len(mst) != m - 1):
        smallest = Edge(None, None, np.inf) # the current smallest edge
        visited.append(v)                   # mark vertex as visited

        # adding edges from the current vertex
        for i in range(m):
            if (A[v, i] != 0 and A[v, i] < mean + threshold*mad):
                edges.append(Edge(v, i, A[v, i]))

        # checking for the smallest edge from the visited vertices to a non-visited vertex
        for edge in edges:
            if (edge.weight < smallest.weight and edge.vTo not in visited):
                smallest = edge

        # remove smallest edge from potential edges and add it to mst
        if (smallest.weight == np.inf):
            print(K)
            K += 1
            clusters[K] = []
            v = 1
            while (v in visited):
                v += 1
                if (v == m):
                    del clusters[K]
                    K -= 1
                    return mst, one, two, distances, K, clusters
        else:
            edges.remove(smallest)
            mst.append(smallest)
            one.append(smallest.vFrom)
            two.append(smallest.vTo)
            distances.append(smallest.weight)
            clusters[K].append([smallest.vFrom, smallest.vTo])
            v = smallest.vTo

    return mst, one, two, distances, K, clusters

# fitting data for alpha vs distance to: dist = -A*(alphas-x0)**2 + y0
def fit(alphas, A, x0, y0):
    return -A*(alphas - x0)**2 + y0

# load the data
data = np.load("clusters.npy")
ra0, dec0 = data['RA'].mean(), data['DEC'].mean()
X = np.dstack(((ra0-data['RA'])*np.cos(np.radians(data['DEC'])), data['DEC']-dec0, data['Z']))[0]  
m = len(data)   # number of data points
print(m, data.dtype.names)

# creating msts for different values of alpha
G = []
alphas = np.linspace(2, 7, 16)    # no change from 0-0.012, much lower values at higher alphas. 2-7
for a in alphas:
    global alpha
    alpha = a
    G.append(kneighbors_graph(X, n_neighbors=100, mode='distance', metric='pyfunc', metric_params={'func':alpha_metric}))

#T = minimum_spanning_tree(G[]).toarray()

count = 0
dist = []
for A in G:
    print("\nAlpha: " + str(alphas[count]))
    lengths = []
    for i in range(m):
        for j in range(i+1, m):
            l = A[i, j]
            if (l != 0):
                lengths.append(l)
    mean = np.mean(lengths)
    mad = stats.median_absolute_deviation(lengths)
    threshold = -1

    # run Prim's algorithm to find clusters
    mst, one, two, distances, K, clusters = prims(A, m, mean, mad, threshold)
    print("Number of clusters: " + str(K))

    # remove the 99 percentile longest edges
    p = 99
    #cuts = np.percentile([T[T > 0]], p)
    #one, two = np.where((T > 0) & (T < cuts))
    #k = m - len(one) - 1    # number of clusters
    
    #plot_mst(one, two, p)

    curr_dist = 0
    for i in clusters:
        a = []
        b = []
        mean_dist = 0
        for j in clusters[i]:
            a.append(j[0])
            b.append(j[1])
            mean_dist += np.sqrt(j[0]**2+j[1]**2)
        mean_dist /= len(clusters[i])
        curr_dist += mean_dist
        # plot_mst(a, b, p)
    curr_dist /= K
    dist.append(curr_dist)
    print("Mean distance: " + str(curr_dist))
    count += 1

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(alphas, dist)
ax1.set_title("How Distance Changes with Alpha")
ax1.set_xlabel("Alpha")
ax1.set_ylabel("Distance")
fig1.tight_layout()

#alpha_min = alphas[dist.index(min(dist))]
#print("\nSmallest distance: " + str(min(dist)))
#print("Corresponding alpha: " + str(alpha_min))

plt.show()
