import dgl
import torch

class CustomGraphDataset(dgl.data.DGLDataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list
        super(CustomGraphDataset, self).__init__(name='custom')

    def __getitem__(self, idx):
        G, node_features, node_labels, train_mask = self.graph_list[idx]
        dgl_G = dgl.from_networkx(G)
        dgl_G.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
        dgl_G.ndata['label'] = torch.tensor(node_labels, dtype=torch.long)
        dgl_G.ndata['train_mask'] = torch.tensor(train_mask)
        return dgl_G

    def __len__(self):
        return len(self.graph_list)