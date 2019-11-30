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
def cluster(A, m, mean, mad, threshold):
    mst = []            # list of edges in the mst
    one = []
    two = []
    edges = []          # list of edges from visited vertices
    v = 0               # starting algorithm from a random vertex
    visited = []        # keeping track of the vertices that have been visited
    K = 1
    clusters = {}       # points in each cluster
    distances = {}      # mean distance between each point in cluster
    clusters[K] = []
    distances[K] = []

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
            if (len(distances[K]) > 1): # ensuring clusters have size greater than 1... good idea? Noise that can be removed? Keep and sort out distance?
                #distances[K] = np.mean(distances[K])
                K += 1
                clusters[K] = []
                distances[K] = []
            #else:   # removing clusters of size 1
            #    del clusters[K]
            #    del distances[K]

            
            v = 1
            while (v in visited):
                v += 1
                if (v == m):
                    del clusters[K]
                    del distances[K]
                    K -= 1
                    return mst, one, two, K, clusters, distances
        else:
            edges.remove(smallest)
            mst.append(smallest)
            one.append(smallest.vFrom)
            two.append(smallest.vTo)
            clusters[K].append([smallest.vFrom, smallest.vTo])
            distances[K].append(smallest.weight)
            v = smallest.vTo

    return mst, one, two, K, clusters, distances
