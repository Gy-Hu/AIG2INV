import dgl
import torch
#import dgl.sparse as dglsp
class CustomGraphDataset(dgl.data.DGLDataset):
    def __init__(self, graph_list, split='train',DIM=128,dual=False):
        self.graph_list = graph_list
        self.split = split
        self.DIM = DIM
        self.dual = dual
        super(CustomGraphDataset, self).__init__(name='custom')
        

    def __getitem__(self, idx):
        G, node_features, node_labels, train_mask = self.graph_list[idx]
        dgl_G = dgl.from_networkx(G)
        #dgl_G = G
        dgl_G.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
        if self.dual: #dual graph -> two features
            # Calculate the index for the split point
            split_point = node_features.shape[1] - self.DIM

            # Divide the features into two parts
            ori_feat = dgl_G.ndata['feat'][:, :split_point]
            struc_feat = dgl_G.ndata['feat'][:, split_point:]

            # Assign the divided features to dgl_G
            dgl_G.ndata['ori_feat'] = ori_feat
            dgl_G.ndata['struc_feat'] = struc_feat
            
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
    
    