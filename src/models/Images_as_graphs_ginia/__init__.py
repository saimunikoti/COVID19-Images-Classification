    import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.datasets as datasets
from torch_geometric.data import InMemoryDataset, Data


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testnset = datasets.MNIST(root='./data', train=False, download=True, transform=None)


class MnistDataset(InMemoryDataset):
    def __init__(self, root, tranform=None, pre_tranform=None):
        super(MnistDataset, self).__init__(root, tranform, pre_tranform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):  # not sure if its implemented correctly.
        return ['raw_data']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process_image(self, image, n):
        counter = 0
        node_attributes = np.zeros(shape=(784, 1))
        for i in range(0, n):
            for j in range(0, n):
                pixel = image.getpixel((i, j))
                node_attributes[counter] = pixel / 255
                counter = counter + 1

        return node_attributes

    def fill_edges(self, G, n):  # method that fills the missing edges.
        for i in range(n):
            for j in range(n):
                if (i == 0):  # up-side
                    if (j == 0):  # exception for up-left node
                        G.add_edge((i, j), (i + 1, j + 1))
                    elif (j == n - 1):  # exception for up-right node
                        G.add_edge((i, j), (i + 1, j - 1))
                    else:
                        G.add_edge((i, j), (i + 1, j + 1))
                        G.add_edge((i, j), (i + 1, j - 1))
                elif (i == n - 1):  # down-side
                    if (j == 0):  # exception for down-left node
                        G.add_edge((i, j), (i - 1, j + 1))
                    elif (j == n - 1):  # exception for down-right node
                        G.add_edge((i, j), (i - 1, j - 1))
                    else:
                        G.add_edge((i, j), (i - 1, j + 1))
                        G.add_edge((i, j), (i - 1, j - 1))
                elif (j == 0):  # left-side
                    G.add_edge((i, j), (i - 1, j + 1))
                    G.add_edge((i, j), (i + 1, j + 1))
                elif (j == n - 1):  # right-side
                    G.add_edge((i, j), (i - 1, j - 1))
                    G.add_edge((i, j), (i + 1, j - 1))
                else:  # middle-nodes
                    G.add_edge((i, j), (i - 1, j - 1))
                    G.add_edge((i, j), (i - 1, j + 1))
                    G.add_edge((i, j), (i + 1, j + 1))
                    G.add_edge((i, j), (i + 1, j - 1))

    def create_graph(self):
        n = 28
        G = nx.grid_2d_graph(n, n)
        graph_list = []
        self.fill_edges(G, n)
        mapping = dict()
        for node in G.nodes():
        	mapping[node] = len(mapping)
        G = nx.relabel_nodes(G, mapping)
        edges = np.array(G.edges()).transpose()
        for x in range(60000):
            image, graph_label = mnist_trainset[x]
            node_attributes = self.process_image(image, n)

            graph = Data(
                x=torch.tensor(node_attributes),
                edge_index=torch.tensor(edges),
                y=graph_label)
            graph_list.append(graph)

        for x in range(10000):
            image, graph_label = mnist_testnset[x]
            node_attributes = self.process_image(image, n)

            graph = Data(
                x=torch.tensor(node_attributes),
                edge_index=torch.tensor(edges),
                y=graph_label)
            graph_list.append(graph)

        return graph_list

    def process(self):
        # Read data into huge `Data` list.
        data_list = self.create_graph()

        # data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
