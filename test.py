import numpy
import networkx as nx
from graph import *
from GUI2 import *
#_graph = Graph(node_count = 300, density=15, directed=False)


DEFAULT_NODECOUNT = 10
DEFAULT_DENSITY = 4
DEFAULT_DIRECTED = False
DEFAULT_SEED = 1

def testGUI(
        node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY, 
        directed = DEFAULT_DIRECTED,
        seed=DEFAULT_SEED):
    gui = GraphUI()
    gui.create_graph_no_GUI(node_count, density, directed, seed)
    gui.render_graph()

def testKruskal(
        node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY,
        seed = DEFAULT_SEED):
    G = Graph(node_count, density, seed=seed)
    G.generateConnectedGraph()
    G.randomizeWeights()
    print(G.kruskal())
    
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
        node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY,
        seed = DEFAULT_SEED):
    gui = GraphUI()
    gui.create_graph_no_GUI(node_count, density, directed=False, seed=seed)

    c = gui.graph.floydWarshall()

    Graph.getShortestPath(5, 1, c)

    gui.render_graph()

def testConnectedComponents(
        node_count = DEFAULT_NODECOUNT, 
        density = DEFAULT_DENSITY,
        seed = DEFAULT_SEED):
    G = Graph(node_count, density, seed=seed)
    G.generateRandomGraph()
    print(f"Number of connected components: {G.countConnectedComponents()}")




# testSorts()
# testWeights()
# testKruskal()
#testFloydWarshall()
testFloydWarshall()


#print(nx.write_network_text(_graph.G))
#print(f"NX Density: {nx.density(_graph.G)}")
#print(f"Degree List: {_graph.G.degree}")
#print(f"Sum of Degrees: {numpy.sum([degree[1] for degree in _graph.G.degree])} | Average Degree: {numpy.average([degree[1] for degree in _graph.G.degree])}")
#print(f"Edges: {_graph.G.number_of_edges()} | Nodes: {_graph.G.number_of_nodes()} | Ratio: {_graph.G.number_of_edges() / _graph.G.number_of_nodes()}")



