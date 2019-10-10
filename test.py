# importing modules 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans, MeanShift
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import kneighbors_graph

# an edge object for the edges in a graph
class Edge():
    '''An edge in a graph'''
    def __init__(self, i, j, weight):
        self.vFrom = i
        self.vTo = j
        self.weight = weight
        self.visited = False 

# running Prim's algorithm to find the mst in a graph G with V vertices
def prims(G, V, mean, mad, threshold):
    mst = []                            # list of edges in the mst
    one = []
    two = []
    edges = []                          # list of edges from visited vertices
    v = 0                               # starting algorithm from a random vertex
    visited = []                        # keeping track of the vertices that have been visited
    distances = []
    n_clusters = 1
    clusters = {}

    # go through vertices until an mst is found
    while(len(mst) != V - 1):
        smallest = Edge(None, None, np.inf) # the current smallest edge
        visited.append(v)                   # mark vertex as visited

        # adding edges from the current vertex
        for i in range(V):
            if (G[v, i] != 0 and G[v, i] < mean + threshold*mad):
                edges.append(Edge(v, i, G[v, i]))

        # checking for the smallest edge from the visited vertices to a non-visited vertex
        for edge in edges:
            if (edge.weight < smallest.weight and edge.vTo not in visited):
                smallest = edge

        # remove smallest edge from potential edges and add it to mst
        if (smallest.weight == np.inf):
            n_clusters += 1
            v = 1
            while (v in visited):
                v += 1
                if (v == V):
                    n_clusters -= 1
                    return mst, one, two, distances, n_clusters
        else:
            edges.remove(smallest)
            mst.append(smallest)
            one.append(smallest.vFrom)
            two.append(smallest.vTo)
            distances.append(smallest.weight)
            v = smallest.vTo

    return mst, one, two, distances, n_clusters
            
# load the data
data = np.load('clusters.npy')
V = len(data)
print(V, data.dtype.names)

# tangent plane to the sky
ra0, dec0 = data['RA'].mean(), data['DEC'].mean()
X = np.dstack(((ra0-data['RA'])*np.cos(data['DEC']*np.pi/180), data['DEC']-dec0))[0]

# graph as a matrix: G[i, j] is the weight of the distance from i to j
G = kneighbors_graph(X, n_neighbors=100, mode='distance')

lengths = []
for i in range(V):
    for j in range(i, V):
        l = G[i, j]
        if (l != 0):
            lengths.append(l)
mean = np.mean(lengths)
mad = stats.median_absolute_deviation(lengths)
threshold = -1.2

# run Prim's algorithm to find clusters
mst, one, two, distances, n_clusters = prims(G, V, mean, mad, threshold)
print(n_clusters)

# plotting mst
fig1 = plt.figure()
ax = fig1.add_subplot(111, aspect='equal')
ax.scatter(X[:,0], X[:,1], alpha=0.1)
plt.quiver(X[one,0], X[one,1], X[two,0]-X[one,0], X[two,1]-X[one,1], angles='xy', scale_units='xy', scale=1, headwidth=0, headaxislength=0, headlength=0, minlength=0)
fig1.tight_layout()

# plotting distances
fig2 = plt.figure()
ax_dist = fig2.add_subplot(111)
x = np.linspace(1, len(distances), len(distances))
ax_dist.plot(x, distances)
ax_dist.set_title("Length of Edges Added")
ax_dist.set_ylabel("Length of added edge")
ax_dist.set_xlabel("Order of edges added")
ax_dist.text(0, 0.8, "Threshold: " + str(threshold) + "\nNumber of Clusters: " + str(n_clusters), transform=ax.transAxes)

plt.show()
