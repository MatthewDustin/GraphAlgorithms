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

        K = self.G.to_directed()
        R = range(K.number_of_nodes())
        pathWeights = [[math.inf for x in R] for a in R]
        nodeChains = [[-1 for x in R] for a in R]
        # arr[:][:] = math.inf

        def printArrArr(array: list[list[float]]):
            for i in range(len(array)):
                for j in range(len(array)):
                    if(array[i][j] == math.inf):
                        print("%5s" % ("IF"), end="")
                    else:
                        print("%5d" % (array[i][j]), end="")
                    if j == len(array)-1:
                        print()        

        for (s, d, w) in K.edges.data("weight"):
            #initialize weights for directly connected nodes
            #mirrors diagonally since we're using an undirected graph
            pathWeights[s][d] = w
            #initialize existing direct paths
            nodeChains[s][d] = s
            nodeChains[d][s] = d
        
        for i in R:
            #weight from a node to itself is 0
            pathWeights[i][i] = 0
            #path from a node to itself is direct
            nodeChains[i][i] = i

        #iterate through potential intermedary nodes
        for inter in R:
            #iterate through all source and destination node combinations
            for src in R:
                for dest in R:
                    #get weight of path from source to intermediary node to destination
                    compoundPath = pathWeights[src][inter] + pathWeights[inter][dest]
                    #if path through intermediary node has a lower weight, update it
                    if pathWeights[src][dest] > compoundPath:
                        #mirrors diagonally since we're using an undirected graph
                        pathWeights[src][dest] = compoundPath
                        #
                        nodeChains[src][dest] = nodeChains[inter][dest]

        printArrArr(pathWeights)
        print()
        printArrArr(nodeChains)

        return nodeChains

    def getShortestPath(start: int, end: int, nodeChains: list[list[float]]):
        print(f"Shortest path from {start} to {end}")
        trace = [end]
        while start != end:
            end = nodeChains[start][end]
            trace.insert(0, end)

        print(trace)
        
        

    def vertexCover(self):
        vertices = set()
        remainingEdges = set(self.G.edges)
        #coveredEdges = set() #Just for tracking, should not be needed for logic
        #coveredNodes = list()

        def uncoveredEdges(node: int):
            uncoveredEdges = [edge for edge in remainingEdges if node in edge]
            return uncoveredEdges
        
        #first adds the incident node with the greater number of uncovered incident edges
        #if the next node still has uncovered edges it is added as well
        def addOptimalNodes(edge: tuple):
            if len(uncoveredEdges(edge[0])) < len(uncoveredEdges(edge[1])): 
                edge = (edge[1], edge[0])
            
            vertices.add(edge[0])
            #coveredNodes.append(edge[0])
            #removeCoveredEdges can be called before including the node because it wont remove anything if the node shouldn't be included
            #avoids including the node when it wouldn't cover any new edges, e.g. leaf nodes
            if removeCoveredEdges(edge[1]) > 0:
                vertices.add(edge[1])
        
        #removes from remainingEdges any edge that is covered by the node provided
        #returns the number of new edges covered
        def removeCoveredEdges(node: int):
            covers = [e for e in remainingEdges if node in e]
            remainingEdges.difference_update(covers)
            #coveredEdges.update(covers)
            #if len(covers) > 0: print(f"removing: {covers}")
            return len(covers)

        while len(remainingEdges) > 0:
            edge = remainingEdges.pop()
            
            #print(edge)

            addOptimalNodes(edge)

            #print(f"edge: {edge} \nremainingEdges: {remainingEdges}")
            #print(f"coveredNodes: {coveredNodes}")            

        print(vertices)

    def countConnectedComponents(self):
        visited = set()
        components = 0

        def dfs(node):
            visited.add(node)
            for neighbor in self.G.neighbors(node):
                if neighbor not in visited:
                    dfs(neighbor)
        
        for node in self.G.nodes:
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
            """Find also performs path compression using the assignment. This makes the tree flat and the find operation O(1) amortized."""
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

        for u, v in self.G.edges:
            union(u, v)

        '''Not all trees are flat, so we repeat the find operation on all nodes'''
        for node in rangeNodeCount:
            root[node] = find(node)

        '''Count all roots. Creating a set is unnecessary.'''
        return sum(1 for node in rangeNodeCount if root[node] == node)
    
