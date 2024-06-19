# https://arxiv.org/pdf/1503.03832.pdf
# https://arxiv.org/pdf/1703.07737.pdf
# http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
# https://github.com/ikonushok/siamese-triplet/tree/master

import warnings

import numpy as np

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch_geometric.data import DataLoader

from trainer import fit
from datasets import SiameseGCN, TripletGCN, BalancedBatchSampler, transform_data_to_graphs, GraphDataset
from metrics import AverageNonzeroTripletsMetric
from networks import EmbeddingNet, ClassificationNet, TripletNet
from losses import TripletLoss, OnlineContrastiveLoss, OnlineTripletLoss

from utils import HardNegativePairSelector  # Strategies for selecting pairs within a minibatch
from utils import RandomNegativeTripletSelector  # Strategies for selecting triplets within a minibatch
from embedding_tools import plot_embeddings, extract_embeddings

from networks import GCNBaseNet, SiameseNet
from losses import ContrastiveLoss


warnings.filterwarnings("ignore")
cuda = torch.cuda.is_available()

# load and Normalize dataset
train_filenames = [['data/dataset/train_2022-06-01_2023-05-31_260d/buy_patterns.npy', '../data/dataset/train_2022-06-01_2023-05-31_260d/buy_patterns_clusters.npy'],
                   ['data/dataset/train_2022-06-01_2023-05-31_260d/sell_patterns.npy', '../data/dataset/train_2022-06-01_2023-05-31_260d/sell_patterns_clusters.npy']]

test_filenames = [['data/dataset/test_2022-06-01_2023-05-31_260d/buy_patterns.npy', '../data/dataset/test_2022-06-01_2023-05-31_260d/buy_patterns_clusters.npy'],
                   ['data/dataset/test_2022-06-01_2023-05-31_260d/sell_patterns.npy', '../data/dataset/test_2022-06-01_2023-05-31_260d/sell_patterns_clusters.npy']]

train_dataset = GraphDataset(transform_data_to_graphs(train_filenames))
test_dataset = GraphDataset(transform_data_to_graphs(test_filenames))

# Common setup
n_classes = 4
batch_size = 64
embedding_dim = 24

# Baseline: Classification with softmax
# We'll train the model for classification and use outputs of penultimate layer as embeddings
# Set up data loaders
title = '1. Baseline - classification with softmax'
print(f'\n{title}:')
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

embedding_net = GCNBaseNet(num_features=10, num_relations=5, embedding_dim=embedding_dim, num_layers=1)
model = ClassificationNet(embedding_net, n_classes=4)

loss_fn = torch.nn.NLLLoss()
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 50
log_interval = 50

train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_baseline, train_labels_baseline, f'{title}, train_embeddings_baseline')
val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_baseline, val_labels_baseline, f'{title}, val_embeddings_baseline')

# While the embeddings look separable (which is what we trained them for),
# they don't have good metric properties. They might not be the best choice as a descriptor for new classes.


# Siamese network
# Now we'll train a siamese network that takes a pair of images and trains the embeddings
# so that the distance between them is minimized if their from the same class or
# greater than some margin value if they represent different classes. We'll minimize a contrastive loss function
# Set up data loaders
title = '2. Siamese network'
print(f'\n{title}:')
siamese_train_dataset = SiameseGCN(train_dataset, train=True)
siamese_test_dataset = SiameseGCN(test_dataset)
batch_size = 64
embedding_dim=64
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
siamese_train_loader = DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
siamese_test_loader = DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
margin = 1.
embedding_net = GCNBaseNet(num_features=10, num_relations=5, embedding_dim=embedding_dim, num_layers=3)
model = SiameseNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 10
log_interval = 100
fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_cl, train_labels_cl, f'{title}, train_embeddings_cl')
val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_cl, val_labels_cl, f'{title}, val_embeddings_cl')


# Triplet network
# We'll train a triplet network, that takes an anchor, positive (same class as anchor)
# and negative (different class than anchor) examples.
# The objective is to learn embeddings such that the anchor is closer to the positive example
# than it is to the negative example by some margin value.
title = '3. Triplet network'
print(f'\n{title}:')

triplet_train_dataset = TripletGCN(train_dataset, train=True) # Returns triplets of images
triplet_test_dataset = TripletGCN(test_dataset)

batch_size = 64
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
margin = 1.
embedding_net = GCNBaseNet(num_features=10, num_relations=5, embedding_dim=embedding_dim, num_layers=3)
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 10
log_interval = 100

fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

train_embeddings_tl, train_labels_tl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_tl, train_labels_tl, f'{title}, train_embeddings')
val_embeddings_tl, val_labels_tl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_tl, val_labels_tl, f'{title}, val_embeddings')


## Online pair selection
## Steps
# 1. Create BalancedBatchSampler - samples N  classes and M samples datasets.py
# 2. Create data loaders with the batch sampler Define embedding (mapping) network f(x) - EmbeddingNet from networks.py
# 3. Define a PairSelector that takes embeddings and original labels and returns valid pairs within a minibatch
# 4. Define OnlineContrastiveLoss that will use a PairSelector and compute ContrastiveLoss on such pairs
# 5. Train the network!
title = '4. Online pair selection - negative mining'
print(f'\n{title}:')
# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=10, n_samples=25)
test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=10, n_samples=25)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# Set up the network and training parameters
margin = 1.
embedding_net = EmbeddingNet()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector())
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50

fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

train_embeddings_ocl, train_labels_ocl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_ocl, train_labels_ocl, f'{title}, train_embeddings')
val_embeddings_ocl, val_labels_ocl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_ocl, val_labels_ocl, f'{title}, val_embeddings')


## Online triplet selection
## Steps
# 1. Create **BalancedBatchSampler** - samples $N$ classes and $M$ samples *datasets.py*
# 2. Create data loaders with the batch sampler
# 3. Define **embedding** *(mapping)* network $f(x)$ - **EmbeddingNet** from *networks.py*
# 4. Define a **TripletSelector** that takes embeddings and original labels and returns valid triplets within a minibatch
# 5. Define **OnlineTripletLoss** that will use a *TripletSelector* and compute *TripletLoss* on such pairs
# 6. Train the network!
title = '5. Online triplet selection - negative mining'
print(f'\n{title}:')
# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=10, n_samples=25)
test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=10, n_samples=25)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# Set up the network and training parameters
margin = 1.
embedding_net = EmbeddingNet()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50

fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
    metrics=[AverageNonzeroTripletsMetric()])

train_embeddings_otl, train_labels_otl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_otl, train_labels_otl, f'{title}, train_embeddings_otl')
val_embeddings_otl, val_labels_otl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_otl, val_labels_otl, f'{title}, val_embeddings_otl')


# display_emb_online, display_emb, display_label_online, display_label = \
#     train_embeddings_otl, train_embeddings_tl, train_labels_otl, train_labels_tl
display_emb_online, display_emb, display_label_online, display_label = \
    val_embeddings_otl, val_embeddings_tl, val_labels_otl, val_labels_tl
x_lim = (np.min(display_emb_online[:,0]), np.max(display_emb_online[:,0]))
y_lim = (np.min(display_emb_online[:,1]), np.max(display_emb_online[:,1]))
plot_embeddings(display_emb, display_label, f'{title}, display_val_emb_otl', x_lim, y_lim)
plot_embeddings(display_emb_online, display_label_online, f'{title}, display_online_val_emb_otl', x_lim, y_lim)

x_lim = (np.min(train_embeddings_ocl[:,0]), np.max(train_embeddings_ocl[:,0]))
y_lim = (np.min(train_embeddings_ocl[:,1]), np.max(train_embeddings_ocl[:,1]))
plot_embeddings(train_embeddings_cl, train_labels_cl, f'{title}, train_embeddings_cl', x_lim, y_lim)
plot_embeddings(train_embeddings_ocl, train_labels_ocl, f'{title}, val_embeddings_cl', x_lim, y_lim)
