from utils.data import load_dataset, generate_pairwise_test, merge_graphs
from utils.sampling import init_sampler, generate_samples
from utils.metrics import *
from model import MultiNetworkEmb
from args import make_args

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
import time


if __name__ == '__main__':
    args = make_args()

    edge_index1, edge_index2, anchor_links, test_links = load_dataset(f'datasets/{args.dataset}', 0.2)
    n1, n2 = edge_index1.max() + 1, edge_index2.max() + 1

    g1, g2 = nx.Graph(), nx.Graph()
    g1.add_nodes_from(np.arange(n1))
    g2.add_nodes_from(np.arange(n2))
    g1.add_edges_from(edge_index1)
    g2.add_edges_from(edge_index2)
    graphs = [g1, g2]

    for g in graphs:
        g.node_sampler = init_sampler(g)
    test_pairs_dict = generate_pairwise_test(test_links)
    id2node, node2id = merge_graphs(graphs, anchor_links)

    # Model
    model = MultiNetworkEmb(num_of_nodes=len(id2node),
                            batch_size=args.batch_size,
                            K=args.neg_samples,
                            node_embedding=args.node_dim,
                            num_layer=len(graphs),
                            layer_embedding=args.layer_dim)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.02, alpha=0.99, eps=1.0, centered=True, momentum=0.0)

    # Training
    for epoch in range(args.epochs):
        # start = time.time()
        train_samples = generate_samples(graphs, args.batch_size, args.neg_samples)
        # print(f'Epoch: {epoch + 1}, time for generating samples: {time.time() - start:.2f}s')

        total_loss = 0
        start = time.time()
        for u_i, u_j, label, gid_vec in train_samples:
            mapped_u_i = np.array([node2id[(gid, u)] for gid, u in zip(gid_vec, u_i)])
            mapped_u_j = np.array([node2id[(gid, u)] for gid, u in zip(gid_vec, u_j)])

            optimizer.zero_grad()
            loss = model(mapped_u_i, mapped_u_j, gid_vec, torch.from_numpy(label))
            loss.backward()
            optimizer.step()
            total_loss += loss

        print(f'Epoch: {epoch + 1}, loss: {total_loss}, time for one epoch: {time.time() - start:.2f}s')
        if epoch % 50 == 49:
            with torch.no_grad():
                embeddings = F.normalize(model.embedding, p=2, dim=1)
                for gid_1 in range(len(graphs)):
                    for gid_2 in range(gid_1 + 1, len(graphs)):
                        embeddings1 = embeddings[[node2id[(gid_1, u)] for u in sorted(graphs[gid_1].nodes)]]
                        embeddings2 = embeddings[[node2id[(gid_2, u)] for u in sorted(graphs[gid_2].nodes)]]
                        similarity = embeddings1 @ embeddings2.T
                        test_pairs = test_pairs_dict[(gid_1, gid_2)]

                        print(f'Graph {gid_1} vs Graph {gid_2} (max): ', end='')
                        hits_ks_max = hits_ks_max_scores(similarity, torch.from_numpy(test_pairs), ks=[1, 5, 10, 30, 50])
                        mrr_max = mrr_max_score(similarity, torch.from_numpy(test_pairs))
                        for k in hits_ks_max:
                            print(f'Hits@{k}: {hits_ks_max[k]:.4f}', end=', ')
                        print(f'Epoch: {epoch + 1}, MRR: {mrr_max:.4f}')

                        print(f'Graph {gid_1} vs Graph {gid_2} (mean): ', end='')
                        hits_ks_mean = hits_ks_mean_scores(similarity, torch.from_numpy(test_pairs), ks=[1, 5, 10, 30, 50])
                        mrr_mean = mrr_mean_score(similarity, torch.from_numpy(test_pairs))
                        for k in hits_ks_mean:
                            print(f'Hits@{k}: {hits_ks_mean[k]:.4f}', end=', ')
                        print(f'Epoch: {epoch + 1}, MRR: {mrr_mean:.4f}')

