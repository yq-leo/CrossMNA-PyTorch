import numpy as np
from collections import defaultdict


def load_dataset(dataset, p):
    """
    Load dataset.
    :param dataset: dataset name
    :param p: training ratio
    :return:
        edge_index1, edge_index2: edge list of graph G1, G2
        anchor_links: training node alignments, i.e., anchor links
        test_pairs: test node alignments
    """

    data = np.load(f'{dataset}_{p:.1f}.npz')
    edge_index1, edge_index2 = data['edge_index1'].T.astype(np.int64), data['edge_index2'].T.astype(np.int64)
    anchor_links, test_pairs = data['pos_pairs'].astype(np.int64), data['test_pairs'].astype(np.int64)

    return edge_index1, edge_index2, anchor_links, test_pairs


def merge_graphs(graphs, anchor_links):
    # TODO: use sparse matrix for anchor_links
    anchor_maps = dict()
    for anchor_link in anchor_links:
        true_anchor = None
        for gid, anchor in enumerate(anchor_link):
            if anchor > -1:
                if true_anchor is None:
                    true_anchor = (gid, anchor)
                else:
                    anchor_maps[(gid, anchor)] = true_anchor

    id2node, node2id = defaultdict(list), dict()
    for gid, g in enumerate(graphs):
        for node in g.nodes():
            if (gid, node) in anchor_maps:
                nid = node2id[anchor_maps[(gid, node)]]
            else:
                nid = len(id2node)
            node2id[(gid, node)] = nid
            id2node[nid].append((gid, node))

    return id2node, node2id


def generate_pairwise_test(test_links):
    test_pairs_dict_list = defaultdict(list)
    for test_link in test_links:
        gids = np.where(test_link > -1)[0]
        for idx1 in range(len(gids)):
            for idx2 in range(idx1 + 1, len(gids)):
                gid1, gid2 = gids[idx1], gids[idx2]
                test_pairs_dict_list[(gid1, gid2)].append([test_link[gid1], test_link[gid2]])

    test_pairs_dict = {}
    for g_pair in test_pairs_dict_list:
        test_pairs_dict[g_pair] = np.array(test_pairs_dict_list[g_pair])

    return test_pairs_dict

