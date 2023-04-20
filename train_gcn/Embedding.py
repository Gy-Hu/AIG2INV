import random
import numpy as np
from gensim.models import Word2Vec
from node2vec import Node2Vec

import numpy as np
import networkx as nx
from sklearn.decomposition import PCA

def deepwalk(G, num_walks, walk_length, embedding_dim, window_size):
    # Perform random walks
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(walk[-1]))
                if not neighbors:
                    break
                walk.append(random.choice(neighbors))
            walks.append([str(n) for n in walk])

    # Train the Skip-gram model
    model = Word2Vec(walks, vector_size=embedding_dim, window=window_size, min_count=0, sg=1, workers=4)

    # Get node embeddings
    node_embeddings = np.zeros((G.number_of_nodes(), embedding_dim))
    for i, node in enumerate(G.nodes()):
        node_embeddings[i, :] = model.wv[str(node)]

    return node_embeddings

'''
node2vec

def generate_node2vec_embedding(graph_data, EMBEDDING_DIM):
    G, _, node_labels, train_mask = graph_data

    # Run Node2Vec on the graph G and obtain the embeddings
    node2vec = Node2Vec(G, dimensions=EMBEDDING_DIM, walk_length=10, num_walks=100, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Assign Node2Vec embeddings as node features
    node_features = np.zeros((G.number_of_nodes(), EMBEDDING_DIM))
    for i, node_id in enumerate(G.nodes()):
        node_features[i] = model.wv[f"{node_id}"]

    return (G, node_features, node_labels, train_mask)
'''

# simple node embedding
def generate_node_features(graph_list, EMBEDDING_DIM):
    pca = PCA(n_components=EMBEDDING_DIM)

    # Compute the centrality measures for the nodes in the graph
    for idx, (G, _, node_labels, train_mask) in enumerate(graph_list):
        betweenness_centrality = nx.betweenness_centrality(G)
        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)

        # Assign the centrality measures as node features
        num_nodes = G.number_of_nodes()
        node_features = np.zeros((num_nodes, 3))
        for i, node_id in enumerate(G.nodes()):
            node_features[i, 0] = betweenness_centrality[node_id]
            node_features[i, 1] = degree_centrality[node_id]
            node_features[i, 2] = closeness_centrality[node_id]
            
        # Apply PCA to increase the dimensionality of the node features
        node_features = pca.fit_transform(node_features)
        graph_list[idx] = (G, node_features, node_labels, train_mask)
        
    return graph_list


# node embedding with attribute -> random embedding
def generate_attribute_node_features(graph_list, EMBEDDING_DIM):
    # Extract unique attributes from all nodes across all graphs
    unique_attributes = set()
    for G, _, _, _ in graph_list:
        for _, node_data in G.nodes(data=True):
            unique_attributes.add(node_data['type'])
            unique_attributes.add(node_data['application'])
            
    # Create an embedding for each unique attribute and store them in a dictionary
    attribute_embedding = {attr: np.random.rand(EMBEDDING_DIM) for attr in unique_attributes}

    # Modify the loop that assigns node features to use the attribute embeddings
    for idx, (G, _, node_labels, train_mask) in enumerate(graph_list):
        node_features = np.zeros((G.number_of_nodes(), EMBEDDING_DIM))
        
        for i, (_, node_data) in enumerate(G.nodes(data=True)):
            node_features[i] = attribute_embedding[node_data['type']] + attribute_embedding[node_data['application']]

        graph_list[idx] = (G, node_features, node_labels, train_mask)

    return graph_list

