import numpy
import networkx as nx
from graph import *
from GUI2 import *


DEFAULT_NODECOUNT = 1000
DEFAULT_DENSITY = 4
DEFAULT_DIRECTED = False
DEFAULT_SEED = None
DEFAULT_GRAPH_GENERATOR = nx.gnm_random_graph

#(nodecound, density, graphGenerator)
TEST_CASES = (
    (10, 1, nx.gnm_random_graph),
    (10, 2, nx.gnm_random_graph),
    (10, 3, nx.gnm_random_graph),
    (100, 10, nx.gnm_random_graph),
    (100, 30, nx.gnm_random_graph),
    (100, 60, nx.gnm_random_graph),
    (1000, 50, nx.gnm_random_graph),
    (1000, 800, nx.gnm_random_graph)
)

def testGUI(
        node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY, 
        directed = DEFAULT_DIRECTED,
        seed=DEFAULT_SEED,
        graphGenerator = DEFAULT_GRAPH_GENERATOR):
    gui = GraphUI()
    gui.create_graph_no_GUI(node_count, density, directed, seed)
    gui.render_graph()
    
def testWeights(
        node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY, 
        directed = DEFAULT_DIRECTED,
        seed = DEFAULT_SEED,
        graphGenerator = DEFAULT_GRAPH_GENERATOR):
    
    G = Graph(node_count, density=density, seed=seed, directed=directed)
    G.randomizeWeights()
    G.randomizeWeights(11, 2)
    # get random edge
    edge = random.choice(list(G.graph.edges))
    print(G.getWeight(edge))
    
    G.setWeight(edge, 5)
    print(G.getWeight(edge))

def testSorts(
        node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY, 
        directed = DEFAULT_DIRECTED,
        seed = DEFAULT_SEED,
        graphGenerator = DEFAULT_GRAPH_GENERATOR):
    G = Graph(node_count, density=density, seed=seed, directed=directed)
    G.randomizeWeights()
    print(G.listWeights())
    print(G.getSortedEdges())

def testFloydWarshallFor(
        node_count = DEFAULT_NODECOUNT,
        density = DEFAULT_DENSITY,
        seed = DEFAULT_SEED,
        graphGenerator = DEFAULT_GRAPH_GENERATOR):
    
    G = Graph(node_count, density=density, graphGenerator=graphGenerator)
    G.randomizeWeights()
    
    nets = nx.floyd_warshall_numpy(G.graph).tolist()
    warshall = G.floydWarshall()[1]
    status = "passed" if warshall != nets else "failed"
    
    print (f"Test for {node_count} nodes and density {density}: {status}")

def testConnectedComponentsFor(
        node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY,
        seed = DEFAULT_SEED,
        graphGenerator = DEFAULT_GRAPH_GENERATOR):
    G = Graph(node_count, density=density, seed=seed)
    numComponents = G.countConnectedComponentsDSU()
    numComponentsNX = nx.number_connected_components(G.graph)
    status = "passed" if numComponentsNX == numComponents else "failed"
    
    print (f"Test for {node_count} nodes and density {density}: {status}")


def testVertexCoverFor(
    node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY,
        seed = DEFAULT_SEED,
        graphGenerator = DEFAULT_GRAPH_GENERATOR):
    
    G = Graph(node_count, density=density, seed=seed)
    coverSize = len(G.vertexCover())
    nxCoverSize = len(G.vertexCoverNX())
    percentDiff = (coverSize - nxCoverSize) / (((nxCoverSize + coverSize)) / 2)
    
    print(f"cover size: {coverSize} networkx cover size: {nxCoverSize} % diff: {percentDiff}")
    


def testKruskalFor(
        node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY,
        seed = DEFAULT_SEED,
        graphGenerator = DEFAULT_GRAPH_GENERATOR):
    
    G = Graph(node_count, density=density, seed=seed, graphGenerator=graphGenerator)
    G.randomizeWeights()
    
    nets= nx.minimum_spanning_tree(G.graph)
    kruskals = G.kruskal()
        
    #get a weird error if I put this directly in the print statement
    s = int(nets.size("weight"))
    print(f"Generated single spanning tree?: {kruskals[0]} | networkx: {s} kruskals: {kruskals[2]}")
    

def testKruskal(seed = DEFAULT_SEED):
    for (n, d, g) in TEST_CASES:
        testKruskalFor(n, d, seed, g)

def testVertexCover(seed = DEFAULT_SEED):
    for (n, d, g) in TEST_CASES:
        testVertexCoverFor(n, d, seed, g)

def testConnectedComponents(seed = DEFAULT_SEED):
    for (n, d, g) in TEST_CASES:
        try:
            testConnectedComponentsFor(n, d, seed, g)
        except RecursionError:
            print("Recursion depth exceeded, can occur for large graphs")
            
def testFloydWarshall(seed = DEFAULT_SEED):
    for (n, d, g) in TEST_CASES:
        testFloydWarshallFor(n, d, seed, g)


# testKruskal()
# testVertexCover()
# testConnectedComponents()
# testFloydWarshall()
