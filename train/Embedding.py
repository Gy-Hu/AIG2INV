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
