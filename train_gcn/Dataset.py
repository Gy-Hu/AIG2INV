import dgl
import torch
#import dgl.sparse as dglsp
class CustomGraphDataset(dgl.data.DGLDataset):
    def __init__(self, graph_list, split='train'):
        self.graph_list = graph_list
        self.split = split
        super(CustomGraphDataset, self).__init__(name='custom')
        

    def __getitem__(self, idx):
        G, node_features, node_labels, train_mask = self.graph_list[idx]
        dgl_G = dgl.from_networkx(G)
        #dgl_G = G
        dgl_G.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
        dgl_G.ndata['label'] = torch.tensor(node_labels, dtype=torch.long)
        dgl_G.ndata['train_mask'] = torch.tensor(train_mask)
        #dgl_G.graph['name'] = G.graph['name']

        # if self.split == 'train':
        #     dgl_G.ndata['val_mask'] = torch.tensor([False] * len(node_labels))
        #     dgl_G.ndata['test_mask'] = torch.tensor([False] * len(node_labels))
        # elif self.split == 'val':
        #     dgl_G.ndata['train_mask'] = torch.tensor([False] * len(node_labels))
        #     dgl_G.ndata['val_mask'] = torch.tensor(train_mask)
        #     dgl_G.ndata['test_mask'] = torch.tensor([False] * len(node_labels))
        # elif self.split == 'test':
        #     dgl_G.ndata['train_mask'] = torch.tensor([False] * len(node_labels))
        #     dgl_G.ndata['val_mask'] = torch.tensor([False] * len(node_labels))
        #     dgl_G.ndata['test_mask'] = torch.tensor(train_mask)
        # else:
        #     raise ValueError("Invalid split value. Allowed values are 'train', 'val', and 'test'.")

        return dgl_G

    def __len__(self):
        return len(self.graph_list)
    
    