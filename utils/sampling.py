import random

import numpy as np


class AliasSampling:
    """Reference: https://en.wikipedia.org/wiki/Alias_method"""
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, num_samples=1):
        x = np.random.rand(num_samples)
        i = np.floor(self.n * x).astype(np.int32)
        y = self.n * x - i
        samples = np.array([i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(num_samples)])
        return samples


def init_sampler(g):
    node_positive_distribution = np.array([d for _, d in sorted(g.degree())], dtype=np.float32) ** 0.75
    node_positive_distribution /= node_positive_distribution.sum()
    node_sampler = AliasSampling(prob=node_positive_distribution)
    return node_sampler


def generate_samples(graphs, batch_size, neg_samples):
    node_samples = []
    for gid, g in enumerate(graphs):
        for a, b in g.edges:
            u_i = np.repeat(a, neg_samples + 1)
            u_j = np.append(b, g.node_sampler.sampling(neg_samples))
            label = np.append(1, np.repeat(-1, neg_samples))
            gid_vec = np.repeat(gid, neg_samples + 1)
            node_samples.append((u_i, u_j, label, gid_vec))
    random.shuffle(node_samples)

    sample_batches = []
    num_batches = len(node_samples) // batch_size
    for i in range(num_batches):
        batch = node_samples[i * batch_size: (i + 1) * batch_size]
        u_i, u_j, label, gid_vec = zip(*batch)
        u_i = np.array(u_i).flatten()
        u_j = np.array(u_j).flatten()
        label = np.array(label).flatten()
        gid_vec = np.array(gid_vec).flatten()
        sample_batches.append((u_i, u_j, label, gid_vec))

    return sample_batches


