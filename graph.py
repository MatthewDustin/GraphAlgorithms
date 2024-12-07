import networkx as nx
import math  
import random

class Graph:
  
    def __init__(self, node_count: int, density=1 , directed=False, seed: int | None = None):
        self.directed = directed
        self.node_count = node_count
        self.seed = seed
        if (self.node_count < 2):
            self.node_count = 2
            print("Cannot have less than 2 nodes, defaulting to two nodes.")
        self.density = density
        
        self.G = None
    
    def generateRandomGraph(self):
    
        self.G = nx.gnm_random_graph(
            n=self.node_count,
            m=self.node_count * self.density * .5,
            seed = self.seed
        )
        
    def generateConnectedGraph(self):
        self.G = nx.DiGraph(weight=0) if self.directed else nx.Graph(weight=0)
        if self.node_count < 1:
            return
        self.G.add_node(0)

        random.seed(self.seed)
        
        for i in range(1, self.node_count):
            #creates list containing a random node key and the node key being added
            temp = random.choices(  
                        list(self.G.nodes),
                        [(w / (i+1)) for w in range(1, i+1)],  #first element's weight should decrease base on how many times it's been in the pool
                        k=1)
            temp.append(i)
            
            #random order only matters when the graph is directed
            if self.directed:
                random.shuffle(temp)
            
            self.G.add_node(i)
            self.G.add_edge(temp[0], temp[1], weight=1)

        target_edge_count = (self.node_count * self.density) * (1 if self.directed else .5)
        
        
        while (self.G.number_of_edges() < target_edge_count):
            temp = random.sample(list(self.G.nodes), 2)
            self.G.add_edge(temp[0], temp[1], weight=1)

    def randomizeWeights(self, minimum=0, offset=10):
        minWeight = min(minimum, minimum+offset)
        maxWeight = max(minimum, minimum+offset)
        random.seed(self.seed)
        for (u, v, w) in self.G.edges.data():
            self.G.edges[u, v]['weight'] = random.randint(minWeight, maxWeight)

    def getWeight(self, edge: tuple):
        return self.G.edges[edge[0], edge[1]]["weight"]
    
    def setWeight(self, edge: tuple, new_weight):
        self.G.edges[edge[0], edge[1]]["weight"] = new_weight

    def listWeights(self):
        return list(self.G.edges.data('weight'))

    def getSortedEdges(self):
        return sorted(self.G.edges, key=self.getWeight)

    def kruskal(self):
            spanningTree = nx.Graph()
            sorted = self.getSortedEdges()

            for (u, v) in sorted:
                if not spanningTree.has_node(u) or not spanningTree.has_node(v):
                    spanningTree.add_edge(u, v)
                    spanningTree.add_node(u)
                    spanningTree.add_node(v)
                if nx.number_of_nodes(spanningTree) == nx.number_of_nodes(self.G):
                    break
            return spanningTree
    
    def floydWarshall(self):
        pathWeights = [[math.inf for x in range(self.node_count)] for a in range(self.node_count)]
        nodeChains = [[-1 for x in range(self.node_count)] for a in range(self.node_count)]
        # arr[:][:] = math.inf

        for (s, d, w) in self.G.edges.data("weight"):
            pathWeights[d][s] = pathWeights[s][d] = w
            nodeChains[s][d] = s
        for i in range(self.node_count):
            pathWeights[i][i] = 0
            nodeChains[i][i] = i

        for inter in range(self.node_count):
            for src in range(self.node_count):
                for dest in range(self.node_count):
                    compoundPath = pathWeights[src][inter] + pathWeights[inter][dest]
                    if pathWeights[src][dest] > compoundPath:
                        pathWeights[src][dest]= compoundPath
                        pathWeights[dest][src]= compoundPath
                        nodeChains[src][dest]= nodeChains[inter][dest]

        Graph.printArrArr(pathWeights)
        print()
        Graph.printArrArr(nodeChains)

        return nodeChains
        
    def printArrArr(array: list[list[float]]):
        for i in range(len(array)):
            for j in range(len(array)):
                if(array[i][j] == math.inf):
                    print("%5s" % ("IF"), end="")
                else:
                    print("%5d" % (array[i][j]), end="")
                if j == len(array)-1:
                    print()        

    def getShortestPath(start: int, end: int, nodeChains: list[list[float]]):
        k = nodeChains[start][end]
        print(start)
        if k==start:
            print(k)
            print(end)
        else:
            Graph.getShortestPath(k, end, nodeChains)


                     
                    
                    
        