import networkx as nx
import numpy as np
import torch
from community_louvain import best_partition
import torchvision.datasets as datasets
from torch_geometric.data import InMemoryDataset, Data

mnist_trainset = datasets.MNIST(root='./data/louvain_avg_pos', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data/louvain_avg_pos', train=False, download=True, transform=None)


class louvainDataset(InMemoryDataset):
    def __init__(self, root, tranform=None, pre_tranform=None):
        super(louvainDataset, self).__init__(root, tranform, pre_tranform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self): 
        return ['raw_data']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process_image(self, image, n):
        counter = 0
        node_attributes = np.zeros(shape=(784, 1))
        position = np.zeros(shape=(784, 2))
        for i in range(n):
            for j in range(n):
                pixel = image.getpixel((i, j))
                node_attributes[counter] = pixel / 255
                position[counter] = np.array([i,j])
                counter = counter + 1

        return node_attributes,position

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

    def compute_weights(self, G, n, node_attributes):
        counter = 0
        for i in range(n * n):
            if (i == 0):
                G.edges[(i), (i + 1)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + 1]))
                G.edges[(i), (i + n)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + n]))
                G.edges[(i), (i + n + 1)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + n + 1]))
            elif (i == n-1):
                G.edges[(i), (i + n)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + n]))
                G.edges[(i), (i + n - 1)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + n + 1]))
            elif(i == (n-1)*n):
                G.edges[(i), (i + 1)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + 1]))
            elif(i == n*n - 1):
                continue
            elif(i < n-1):
                G.edges[(i), (i + 1)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + 1]))
                G.edges[(i), (i + n)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + n]))
                G.edges[(i), (i + n - 1)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + n - 1]))
                G.edges[(i), (i + n + 1)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + n + 1]))
            elif(i%n == 0):
                G.edges[(i), (i + 1)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + 1]))
                G.edges[(i), (i + n)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + n]))
                G.edges[(i), (i + n + 1)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + n + 1]))
            elif((i+1)%n == 0):
                G.edges[(i), (i + n)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + n]))
                G.edges[(i), (i + n - 1)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + n - 1]))
            elif(i > (n-1)*n):
                G.edges[(i), (i + 1)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + 1]))
            else:
                G.edges[(i), (i + 1)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + 1]))
                G.edges[(i), (i + n)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + n]))
                G.edges[(i), (i + n - 1)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + n - 1]))
                G.edges[(i), (i + n + 1)]['weight'] = float(1 - np.abs(node_attributes[i] - node_attributes[i + n + 1]))

          counter = counter + 1

    def compute_atrributes(self,H,no_clusters,partition,node_attributes,position):

        #print(no_clusters)
        new_attributes = np.zeros((no_clusters, 3))
        for cluster in range(no_clusters):
            H.add_node(cluster)
            cluster_nodes = 0
            attributes = 0
            pos_i = 0
            pos_j = 0
            for node in partition:
                #print(partition[cluster])
                if(partition[node] == cluster):
                    cluster_nodes = cluster_nodes + 1
                    attributes = attributes + node_attributes[node]
                    pos_i = pos_i + position[node][0]
                    pos_j = pos_j + position[node][1]

            new_attributes[cluster][0] = float(attributes / cluster_nodes)
            new_attributes[cluster][1] = float(pos_i / (cluster_nodes*27))
            new_attributes[cluster][2] = float(pos_j / (cluster_nodes*27))
            #print(new_attributes)
        return new_attributes

    def create_graph(self):
        n = 28
        G = nx.grid_2d_graph(n, n)
        
        graph_list = []
        self.fill_edges(G, n)
        mapping = dict()
        for node in G.nodes():
            mapping[node] = len(mapping)
        G = nx.relabel_nodes(G, mapping)
        
        for x in range(10000):
            H = nx.Graph()
            image, graph_label = mnist_trainset[x]
            node_attributes,position = self.process_image(image, n)
            #print(node_attributes)
            #print(G.number_of_edges())
            #self.compute_weights(G,n,node_attributes)
            #print(G.edges.data())
            #print(G.number_of_edges())
            partition = best_partition(G)
            print(partition)
            no_clusters = max(partition.values()) + 1
            print(no_clusters)
            new_attributes = self.compute_atrributes(H,no_clusters,partition,node_attributes,position)
            #print(new_attributes)           
            #print(x)
            #print(H.number_of_nodes())
            #print(H.number_of_edges())
            nodes = list(G.nodes())
            #print(nodes)
            edge_matrix = np.zeros(shape=(no_clusters,no_clusters))
            #count_edges = 0
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    if G.has_edge(nodes[i], nodes[j]):
                        if(partition[i]!=partition[j]):
                            H.add_edge(partition[i],partition[j])
                            edge_matrix[partition[i],partition[j]] = edge_matrix[partition[i],partition[j]] + 1
                            #count_edges = count_edges + 1
            
            #print(count_edges)
           
            edge_attributes = np.zeros(H.number_of_edges())
            counter = 0 
            for i in range(no_clusters):
                for j in range(no_clusters):
                    if(edge_matrix[i,j] != 0 or edge_matrix[j,i]!=0):
                        edge_attributes[counter] = edge_matrix[i,j] + edge_matrix[j,i]
                        edge_matrix[i,j] = 0
                        edge_matrix[j,i] = 0
                        counter = counter + 1
                    
           
            edges = np.array(H.edges()).transpose()
            graph = Data(
                x=torch.tensor(new_attributes),
                edge_index=torch.tensor(edges),
                edge_attr=torch.tensor(edge_attributes), 
                y=graph_label)
            graph_list.append(graph)

        for x in range(1000):
            H = nx.Graph()
            image, graph_label = mnist_testset[x]
            node_attributes,position = self.process_image(image, n)
            self.compute_weights(G,n,node_attributes)
            partition = best_partition(G)
           
            no_clusters = max(partition.values()) + 1
            new_attributes = self.compute_atrributes(H,no_clusters,partition,node_attributes,position)
            #print(new_attributes)
            nodes = list(G.nodes())
            
            edge_matrix = np.zeros(shape=(no_clusters,no_clusters))
            #count_edges = 0
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    if G.has_edge(nodes[i], nodes[j]):
                        if(partition[i]!=partition[j]):
                            H.add_edge(partition[i],partition[j])
                            edge_matrix[partition[i],partition[j]] = edge_matrix[partition[i],partition[j]] + 1

            edge_attributes = np.zeros(H.number_of_edges())
            counter = 0 
            for i in range(no_clusters):
                for j in range(no_clusters):
                    if(edge_matrix[i,j] != 0 or edge_matrix[j,i]!=0):
                        edge_attributes[counter] = edge_matrix[i,j] + edge_matrix[j,i]
                        edge_matrix[i,j] = 0
                        edge_matrix[j,i] = 0
                        counter = counter + 1
            
            edges = np.array(H.edges()).transpose()

            graph = Data(
                x=torch.tensor(new_attributes),
                edge_index=torch.tensor(edges),
                edge_attr=torch.tensor(edge_attributes),
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
