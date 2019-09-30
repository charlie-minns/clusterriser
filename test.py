# importing modules 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
def prims(G, V):
    mst = []                            # list of edges in the mst
    one = []
    two = []
    edges = []                          # list of edges from visited vertices
    v = 0                               # starting algorithm from a random vertex
    visited = []                        # keeping track of the vertices that have been visited
    distances = []

    # go through vertices until an mst is found
    while(len(mst) != V - 1):
        smallest = Edge(None, None, np.inf) # the current smallest edge
        visited.append(v)                   # mark vertex as visited

        # adding edges from the current vertex
        for i in range(V):
            if (G[v, i] != 0):
                edges.append(Edge(v, i, G[v, i]))

        # checking for the smallest edge from the visited vertices to a non-visited vertex
        for edge in edges:
            if (edge.weight < smallest.weight and edge.vTo not in visited):
                smallest = edge

        # remove smallest edge from potential edges and add it to mst
        if (smallest.weight == np.inf):
            v = 1
            while (v in visited):
                v += 1
                if (v == V):
                    return mst, one, two
        else:
            edges.remove(smallest)
            mst.append(smallest)
            one.append(smallest.vFrom)
            two.append(smallest.vTo)
            distances.append(smallest.weight)
            v = smallest.vTo

    return mst, one, two
            
# load the data
data = np.load('clusters.npy')
V = len(data)
print(V, data.dtype.names)

# tangent plane to the sky
ra0, dec0 = data['RA'].mean(), data['DEC'].mean()
X = np.dstack(((ra0-data['RA'])*np.cos(data['DEC']*np.pi/180), data['DEC']-dec0))[0]

# graph as a matrix: G[i, j] is the weight of the distance from i to j
G = kneighbors_graph(X, n_neighbors=100, mode='distance')

# run Prim's algorithm
mst, one, two = prims(G, V)

# plotting
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.scatter(X[:,0], X[:,1], alpha=0.1)
plt.quiver(X[one,0], X[one,1], X[two,0]-X[one,0], X[two,1]-X[one,1], angles='xy', scale_units='xy', scale=1, headwidth=0, headaxislength=0, headlength=0, minlength=0)
fig.tight_layout()
plt.show()
