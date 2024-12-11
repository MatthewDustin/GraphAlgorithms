import numpy
import networkx as nx
from graph import *
from GUI2 import *
#_graph = Graph(node_count = 300, density=15, directed=False)


DEFAULT_NODECOUNT = 1000
DEFAULT_DENSITY = 4
DEFAULT_DIRECTED = False
DEFAULT_SEED = None

def testGUI(
        node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY, 
        directed = DEFAULT_DIRECTED,
        seed=DEFAULT_SEED):
    gui = GraphUI()
    gui.create_graph_no_GUI(node_count, density, directed, seed)
    gui.render_graph()
    
def testWeights(
        node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY, 
        directed = DEFAULT_DIRECTED,
        seed = DEFAULT_SEED):
    
    G = Graph(node_count, directed, density, seed)
    G.generateConnectedGraph()
    G.randomizeWeights()
    G.randomizeWeights(11, -2)
    print(G.listWeights()[0])
    edge = G.listWeights()[0]
    print(G.getWeight(edge))
    G.setWeight(edge, 5)
    print(G.getWeight(edge))

def testSorts(
        node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY, 
        directed = DEFAULT_DIRECTED,
        seed = DEFAULT_SEED):
    G = Graph(node_count, directed, density)
    G.randomizeWeights()
    print(G.listWeights())
    print(G.getSortedEdges())

def testFloydWarshall(
        density = DEFAULT_DENSITY,
        seed = DEFAULT_SEED):
    
    def testForNodeCount(node_count):
        G = Graph(node_count, density)
        G.generateConnectedGraph()
        nets = nx.floyd_warshall_numpy(G.G).tolist()
        warshall = G.floydWarshall()[1]
        if warshall != nets: print(f"Failed with node_cout = {node_count}")

    # for i in range(2, 9):
    #     testForNodeCount(i)
    for i in range(10, 1000, 100):
        testForNodeCount(i)

    print("Finished testFloydWarshall")
    

def testConnectedComponents(
        node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY,
        seed = DEFAULT_SEED):
    G = Graph(node_count, density, seed=seed)
    G.generateRandomGraph()
    print(f"Number of connected components: {nx.number_connected_components(G.G)}")
    print(f"Number of connected components: {G.countConnectedComponents()}")


def testVertexCover(
    node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY,
        seed = DEFAULT_SEED):
    G = Graph(node_count, density, seed=seed)
    G.generateRandomGraph()

    cover = G.vertexCover()
    if not nx.algorithms.is_vertex_cover(G.G, cover): 
        print("FAILED: Not a vertex cover")
        return


def testKruskal(
        node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY,
        seed = DEFAULT_SEED):
    
    G = Graph(node_count, density, seed=seed)
    G.generateConnectedGraph()
    G.randomizeWeights()
    nets= nx.minimum_spanning_tree(G.G)
    kruskals = G.kruskal()
    print(G.countConnectedComponentsNX())
    #get a weird error if I put this directly in the print statement
    s = int(nets.size("weight"))
    print(f"Single spanning tree exists?: {kruskals[0]} networkx: {s} kruskals: {kruskals[2]}")




# testConnectedComponents()
# testSorts()
# testWeights()
testKruskal(density=1)
# testFloydWarshall()
# testVertexCover(seed = None)


#print(nx.write_network_text(_graph.G))
#print(f"NX Density: {nx.density(_graph.G)}")
#print(f"Degree List: {_graph.G.degree}")
#print(f"Sum of Degrees: {numpy.sum([degree[1] for degree in _graph.G.degree])} | Average Degree: {numpy.average([degree[1] for degree in _graph.G.degree])}")
#print(f"Edges: {_graph.G.number_of_edges()} | Nodes: {_graph.G.number_of_nodes()} | Ratio: {_graph.G.number_of_edges() / _graph.G.number_of_nodes()}")



