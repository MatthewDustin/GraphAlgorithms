import tkinter as tk
from tkinter import messagebox, simpledialog
import random
import matplotlib.pyplot as plt
import mplcursors
from graph import *


class GraphUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Graph Operations")
        self.geometry("600x400")

        self.graph = None

        self.create_widgets()

    def create_widgets(self):
        # Create buttons for different actions
        tk.Button(self, text="Create Graph", command=self.create_graph).pack(pady=5)
        tk.Button(self, text="Add Edge", command=self.add_edge).pack(pady=5)
        tk.Button(self, text="Delete Edge", command=self.delete_edge).pack(pady=5)
        tk.Button(self, text="Edit Weight", command=self.edit_weight).pack(pady=5)
        tk.Button(self, text="Render Graph", command=self.render_graph).pack(pady=5)
        tk.Button(self, text="Shortest Path", command = self.get_shortest_path).pack(pady=5)
        tk.Button(self, text="Exit", command=self.quit).pack(pady=5)

        self.output_area = tk.Text(self, height=15, width=70)
        self.output_area.pack(pady=10)

    def create_graph(self):
        node_count = simpledialog.askinteger("Input", "Enter number of nodes:")
        density = simpledialog.askfloat("Input", "Enter average degree (density):")
        directed = simpledialog.askstring("Input", "Directed graph? (y/n):").lower() == 'y'
        random_weights = simpledialog.askstring("Input", "Random weights? (y/n):").lower() == 'y'

        if node_count is None or density is None:
            return

        self.graph = Graph(node_count, density, directed)
        self.graph.generateRandomGraph()
        if random_weights: 
            self.graph.randomizeWeights()

    def create_graph_no_GUI(self, node_count, density, directed=False, seed: int | None = None):

        self.graph = Graph(node_count, density, directed, seed)
        self.graph.generateRandomGraph()
        self.graph.randomizeWeights()


    def add_edge(self):
        if self.graph is None:
            messagebox.showerror("Error", "Please create a graph first.")
            return
        
        u = simpledialog.askinteger("Input", "Enter starting node (u):")
        v = simpledialog.askinteger("Input", "Enter ending node (v):")
        weight = simpledialog.askfloat("Input", "Enter weight:")
        
        if u is not None and v is not None and weight is not None:
            self.graph.G.add_edge(u, v, weight)
            self.output_area.insert(tk.END, f"Edge added: {u} -- {v} (Weight: {weight})\n")

    def get_shortest_path(self):
        if self.graph is None:
            messagebox.showerror("Error", "Please create a graph first.")
            return
        
        u = simpledialog.askinteger("Input", "Enter starting node (u):")
        v = simpledialog.askinteger("Input", "Enter ending node (v):")

        if u is not None and v is not None:
            warshall = self.graph.floydWarshall()
            Graph.getShortestPath(u, v, warshall[0])

    def delete_edge(self):
        if self.graph is None:
            messagebox.showerror("Error", "Please create a graph first.")
            return

        u = simpledialog.askinteger("Input", "Enter starting node (u):")
        v = simpledialog.askinteger("Input", "Enter ending node (v):")

        if u is not None and v is not None:
            self.graph.G.remove_edge(u, v)
            self.output_area.insert(tk.END, f"Edge deleted: {u} -- {v}\n")

    def edit_weight(self):
        if self.graph is None:
            messagebox.showerror("Error", "Please create a graph first.")
            return

        u = simpledialog.askinteger("Input", "Enter starting node (u):")
        v = simpledialog.askinteger("Input", "Enter ending node (v):")
        new_weight = simpledialog.askfloat("Input", "Enter new weight:")

        if u is not None and v is not None and new_weight is not None:
            self.graph.edit_weight(u, v, new_weight)
            self.output_area.insert(tk.END, f"Weight of edge {u} -- {v} updated to {new_weight}.\n")

    def render_graph(self):
        if self.graph is None:
            messagebox.showerror("Error", "Please create a graph first.")
            return


        pos = nx.spring_layout(self.graph.G)  # Positions for all nodes

        # Extract weights for color mapping
        weights = [self.graph.G[u][v]['weight'] for u, v in self.graph.G.edges()]

        # Normalize weights for color mapping
        min_weight = min(weights)
        max_weight = max(weights)
        diff = (max_weight - min_weight)
        diff = 1 if diff == 0 else diff
        #avoids error when all weights equal

        normal_weights = [(w - min_weight) / diff for w in weights]

        # Create a color map
        cmap = plt.get_cmap('coolwarm')  # Blue to red colormap
        edge_colors = [cmap(norm) for norm in normal_weights]  

        fig, ax = plt.subplots()

        nx.draw(self.graph.G, pos, with_labels=True, node_size=700, node_color='lightblue', ax=ax)
        edges = nx.draw_networkx_edges(self.graph.G, pos, edge_color=edge_colors, width=2, ax=ax)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Edge Weight')

        edge_labels = [(u, v) for u, v in self.graph.G.edges()]
        cursor = mplcursors.cursor(edges, hover=2)

        cursor.connect("add", lambda sel: sel.annotation.set_text(
            f"Weight: {self.graph.G[edge_labels[sel.index[0]][0]][edge_labels[sel.index[0]][1]]['weight']}\nEdge: {edge_labels[sel.index[0]][0]} -- {edge_labels[sel.index[0]][1]}"
        ))

        # Clear tooltip when not hovering
        cursor.connect("remove", lambda sel: sel.annotation.set_visible(False))

        plt.title("Graph Visualization")
        plt.show()


if __name__ == "__main__":
    app = GraphUI()
    app.mainloop()
