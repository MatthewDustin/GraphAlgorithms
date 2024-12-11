import networkx as nx
import math  
import random
import time
import heapq
from typing import Callable


import numpy as np

class Graph:
    def __init__(self, node_count: int, 
                density=1,
                directed=False, 
                seed: int | None = None, 
                graphGenerator : Callable[[int, int, int, bool], any] = nx.gnm_random_graph):
        self.graphGenerator = graphGenerator
        self.directed = directed
        self.node_count = node_count
        self.seed = seed
        if (self.node_count < 2):
            self.node_count = 2
            print("Cannot have less than 2 nodes, defaulting to two nodes.")
        self.density = density
        if (density > self.node_count - 1):
            print("Too many edges for given vertice count")
            self.density = node_count - 1
        
        self.graph : nx.Graph | nx.DiGraph
        self.generate()

    def copy(self): 
        newGraph = self.graph.copy()
        graph = Graph(self.node_count, self.density, self.directed, seed=self.seed)
        graph.graph = newGraph
        return graph
    
    def generate(self):
        edge_count = (self.node_count * self.density) * (1 if self.directed else .5)
        self.graph = self.graphGenerator(
            self.node_count,
            edge_count, 
            self.seed,
            self.directed)
        

    def randomizeWeights(self, minimum=0, offset=10):
        random.seed(self.seed)
        for (u, v, w) in self.graph.edges.data():
            self.graph.edges[u, v]['weight'] = random.randint(minimum, minimum + offset)

    def getWeight(self, edge: tuple):
        return self.graph.edges[edge[0], edge[1]]["weight"]
    
    def setWeight(self, edge: tuple, new_weight):
        self.graph.edges[edge[0], edge[1]]["weight"] = new_weight

    def listWeights(self):
        return list(self.graph.edges.data('weight'))

    def getSortedEdges(self):
        return sorted(self.graph.edges.data('weight'), key=self.getWeight)

    def kruskal(self):
            spanningEdges = []
            clusters = {node: {node} for node in self.graph.nodes}
            queue = self.getSortedEdges()
            totalWeight = 0

            while len(clusters[1]) < self.graph.number_of_nodes():
                #if the queue is empty then no single spanning tree exists
                if not queue: return (False, spanningEdges, totalWeight)
                (u, v, w) = queue.pop(0)
                
                if clusters[u] != clusters[v]:
                    spanningEdges.append((u, v, w))
                    totalWeight += w
                    
                    c = clusters[u] 
                    c.update(clusters[v])
                    # print(f"Merging {u}:{clusters[u]} and {v}:{clusters[v]}")
                    # print(f"into: {c}")
                    clusters.update(dict.fromkeys(c, c))

                
            return (True, spanningEdges, totalWeight)
    
    
    def kruskalNX(self):
        return nx.minimum_spanning_tree(self.graph)
    
    def floydWarshall(self):
        # Get nodes and initialize indices
        num_nodes = len(self.graph)
        node_indices = {node: idx for idx, node in enumerate(self.graph)}

        pathWeights = [[math.inf] * num_nodes for _ in range(num_nodes)]
        nodeChains = [[-1] * num_nodes for _ in range(num_nodes)]

        # Set weights from edges
        for s, d, w in self.graph.edges.data("weight", default=math.inf):
            s_idx, d_idx = node_indices[s], node_indices[d]
            pathWeights[s_idx][d_idx] = w
            nodeChains[s_idx][d_idx] = s_idx

        for i in range(num_nodes):
            pathWeights[i][i] = 0
            nodeChains[i][i] = i

        # Floyd-Warshall algorithm
        for intermediarry in range(num_nodes):
            for source in range(num_nodes):
                srcToInt = pathWeights[source][intermediarry]
                for destination in range(num_nodes):
                    compoundPath =  srcToInt + pathWeights[intermediarry][destination]
                    if pathWeights[source][destination] > compoundPath:
                        pathWeights[source][destination] = compoundPath
                        nodeChains[source][destination] = nodeChains[intermediarry][destination]

        return nodeChains, pathWeights

    def floysWarshallNX(self):
        return nx.floyd_warshall(self.graph, weight='weight')

    def getShortestPath(start: int, end: int, nodeChains: list[list[float]]):
        #print(f"Shortest path from {start} to {end}")
        trace = [end]
        while start != end:
            end = nodeChains[start][end]
            trace.insert(0, end)

        return trace
    
    def vertexCover(self):
        vertices = set()
        remainingEdges = set(self.graph.edges)
        connectedEdges = {}
        
        for (u, v) in self.graph.edges:
            connectedEdges.update({u: (u, v)})
            connectedEdges.update({v: (u, v)}) 

        while len(remainingEdges) > 0:
            (u, v) = remainingEdges.pop()
            
            vertices.add(u)
            remainingEdges.difference_update(connectedEdges[u])
            vertices.add(v)
            remainingEdges.difference_update(connectedEdges[v])



            #print(f"edge: {edge} \nremainingEdges: {remainingEdges}")
            #print(f"coveredNodes: {coveredNodes}")            
        return vertices
    
    def vertexCoverNX(self):
        return nx.algorithms.approximation.min_weighted_vertex_cover(self.graph)
    
    def countConnectedComponents(self):
        visited = set()
        components = 0

        def dfs(node):
            visited.add(node)
            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    dfs(neighbor)
        
        for node in self.graph.nodes:
            if node not in visited:
                components += 1
                dfs(node)
        
        return components

    def countConnectedComponentsDSU(self):
        """This algorithm uses Disjoint Set Union. Each node starts as a tree root."""
        rangeNodeCount = range(self.node_count)
        root = list(rangeNodeCount)
        depth = [0] * self.node_count

        def find(node):
            if root[node] != node:
                root[node] = find(root[node])
            return root[node]

        def union(node1, node2):
            root1 = find(node1)
            root2 = find(node2)

            if root1 != root2:
                '''Attach the smaller tree to the root of the larger tree'''
                if depth[root1] > depth[root2]:
                    root[root2] = root1
                elif depth[root1] < depth[root2]:
                    root[root1] = root2
                else:
                    depth[root1] += 1
                    root[root2] = root1

        for u, v in self.graph.edges:
            union(u, v)

        '''Not all trees are flat, so we repeat the find operation on all nodes'''
        for node in rangeNodeCount:
            root[node] = find(node)

        '''Count all roots. Creating a set is unnecessary.'''
        return sum(1 for node in rangeNodeCount if root[node] == node)
    
    def countConnectedComponentsNX(self):
        return nx.number_connected_components(self.graph)

'''
start_time = time.time()
for i in range(1000):
    graph = Graph(75, density=3, seed=(i + 37), directed=False)
    graph.randomizeWeights()
    graph.floydWarshall()
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
for i in range(1000):
    graph = Graph(75, density=3, seed=(i + 37), directed=False)
    graph.randomizeWeights()
    graph.floysWarshallNX()
print("--- %s seconds ---" % (time.time() - start_time))
'''   

    # def generateConnectedGraph(self):
    #     self.graph = nx.DiGraph(weight=0) if self.directed else nx.Graph(weight=0)
    #     if self.node_count < 1:
    #         return
    #     self.G.add_node(0)

    #     random.seed(self.seed)
        
    #     for i in range(1, self.node_count):
    #         #creates list containing a random node key and the node key being added
    #         temp = random.choices(  
    #                     list(self.G.nodes),
    #                     [(w / (i+1)) for w in range(1, i+1)],  #first element's weight should decrease base on how many times it's been in the pool
    #                     k=1)
    #         temp.append(i)
            
    #         #random order only matters when the graph is directed
    #         if self.directed:
    #             random.shuffle(temp)
            
    #         self.G.add_node(i)
    #         self.G.add_edge(temp[0], temp[1], weight=1)

    #     target_edge_count = (self.node_count * self.density) * (1 if self.directed else .5)
        
        
    #     while (self.G.number_of_edges() < target_edge_count):
    #         temp = random.sample(list(self.G.nodes), 2)
    #         self.G.add_edge(temp[0], temp[1], weight=1)
            
    # def generateRandomGraph(self):
    
    #     self.G = nx.gnm_random_graph(
    #         n=self.node_count,
    #         m=self.node_count * self.density * .5,
    #         seed = self.seed
    #     )
