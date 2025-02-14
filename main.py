import torch
import time
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from AUC import *
from align_metric import *
from model import MultiNetworkEmb

random.seed(0)
np.random.seed(0)

# Argument parser setup
parser = ArgumentParser("network alignment", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument('--task', default='NetworkAlignment', type=str)
parser.add_argument("--p", default=0.5, type=str)
parser.add_argument('--dataset', default='Twitter', type=str)
parser.add_argument('--node-dim', default=200, type=int, help='d1')
parser.add_argument('--layer-dim', default=100, type=int, help='d2')
parser.add_argument('--batch-size', default=512 * 8, type=int)
parser.add_argument('--neg-samples', default=1, type=int)
parser.add_argument('--output', default='node2vec.pk', type=str)
parser.add_argument('--epochs', default=400, type=int)
args = parser.parse_args()

# Print selected arguments
dataset = args.dataset
p = args.p
print("Dataset:", dataset, "p:", p, "Task:", args.task)

# Determine path based on task
if args.task == 'NetworkAlignment':
    path = f'node_matching/{dataset}/new_network{p}.txt'
elif args.task == 'LinkPrediction':
    path = f'link_prediction/{dataset}/train{p}.txt'

# Step 1: Load data
layers, num_nodes, id2node = readfile(graph_path=path)
num_layers = len(layers.keys())

# Step 2: Initialize negative sampling table
for layerid in layers:
    g = layers[layerid]
    g.init_neg()

# Step 3: Create model
model = MultiNetworkEmb(num_of_nodes=num_nodes,
                        batch_size=args.batch_size,
                        K=args.neg_samples,
                        node_embedding=args.node_dim,
                        num_layer=num_layers,
                        layer_embedding=args.layer_dim)

# Step 4: Training Loop
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.025, alpha=0.99, eps=1.0, centered=True, momentum=0.0)

for epoch in range(args.epochs):  # Example training loop
    t = time.time()
    batches = gen_batches(layers, batch_size=args.batch_size, K=args.neg_samples)
    print("epoch {0}: time for generate batches={1}s".format(epoch, time.time() - t))

    total_loss = 0
    t = time.time()
    for batch in batches:
        u_i, u_j, label, this_layer = batch
        u_i = torch.from_numpy(u_i)
        u_j = torch.from_numpy(u_j)
        label = torch.from_numpy(label)
        this_layer = torch.from_numpy(this_layer).squeeze()

        optimizer.zero_grad()
        loss = model(u_i, u_j, this_layer, label)
        loss.backward()
        optimizer.step()
        total_loss += loss
    print("epoch {0}: loss={1}, time for one epoch={2}s".format(epoch, total_loss, time.time() - t))

    if epoch % 5 == 0 and epoch > 1:
        if args.task == 'NetworkAlignment':
            with torch.no_grad():
                inter_vectors = model.embedding.cpu().numpy()
            node2vec = get_alignment_emb(inter_vectors, layers, id2node)
            result = eval_emb(f'node_matching/{args.dataset}/networks{args.p}.pk', node2vec)
            start_time = time.time()
            pickle.dump(inter_vectors, open(f'emb/{args.output}', 'wb'))
            end_time = time.time()
            print(f'epoch {epoch}: time for alignment {end_time - start_time}')
        elif args.task == 'LinkPrediction':
            with torch.no_grad():
                inter_vectors = model.embedding.cpu().numpy()
                W = model.W.cpu().numpy()
                layers_embedding = model.L_embedding.cpu().numpy()
            node2vec = get_intra_emb(inter_vectors, W, layers_embedding, layers, id2node)
            auc = []
            for i in range(1, num_layers + 1):
                each_auc = [AUC(node2vec[i], f'link_prediction/{dataset}/test{p}.txt', i) for _ in range(5)]
                auc.append(np.mean([x for x in each_auc if x]))
            print(f'epoch {epoch}: auc={np.mean(auc)}')
            pickle.dump([inter_vectors, W, layers_embedding], open(f'emb/{args.output}', 'wb'))

print("Training Complete.")
