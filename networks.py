import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class GCNBaseNet(nn.Module):
    def __init__(self, num_features, num_relations, embedding_dim, num_layers):
        super(GCNBaseNet, self).__init__()
        self.num_relations = num_relations
        self.gcn_layers = nn.ModuleList([
            nn.ModuleDict({
                f'gcn_{i}': GCNConv(embedding_dim if layer > 0 else num_features, embedding_dim)
                for i in range(num_relations)
            }) for layer in range(num_layers)
        ])

        self.fc_int = nn.Sequential(nn.Linear(embedding_dim * num_relations, embedding_dim),
                                    nn.ReLU(),
                                    nn.Linear(embedding_dim, embedding_dim))

        self.fc = nn.Sequential(
            nn.Linear(60 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 2)
        )

    def forward(self, data_list):
        current_embeddings = [data_list[i].x for i in range(len(data_list))]  # !!!
        for layer in self.gcn_layers:
            next_embeddings = []
            for i in range(len(data_list)):  # iteration over batch - could be significantly speed up
                acc_emb = []
                for j in range(self.num_relations):
                    x, edge_index, edge_weight = current_embeddings[i], data_list[i][f'edge_index_type_{j}'], data_list[i][f'edge_attr_type_{j}']
                    emb = layer[f'gcn_{j}'](x, edge_index, edge_weight=edge_weight)  # Apply GCN layer
                    acc_emb.append(emb)
                acc = torch.cat(acc_emb, dim=1)
                acc = self.fc_int(acc)
                next_embeddings.append(acc)
            current_embeddings = next_embeddings
        current_embeddings = torch.stack(
            current_embeddings)  # I want to keep common embeddings and use interconnectivity
        current_embeddings = current_embeddings.view(current_embeddings.size(0), -1)
        output = self.fc(current_embeddings)
        return output

    def get_embedding(self, data_list):
        return self.forward(data_list)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
