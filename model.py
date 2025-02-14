import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiNetworkEmb(nn.Module):
    def __init__(self, num_of_nodes, batch_size, K, node_embedding, num_layer, layer_embedding):
        super(MultiNetworkEmb, self).__init__()
        self.batch_size = batch_size
        self.K = K

        # Parameters (Replaces TensorFlow's tf.Variable)
        self.embedding = nn.Parameter(torch.empty(num_of_nodes, node_embedding))
        self.L_embedding = nn.Parameter(torch.empty(num_layer + 1, layer_embedding))
        self.W = nn.Parameter(torch.empty(node_embedding, layer_embedding))

        # Initialize with truncated normal (approximation using normal distribution)
        nn.init.trunc_normal_(self.embedding, mean=0.0, std=0.3)
        nn.init.trunc_normal_(self.L_embedding, mean=0.0, std=0.3)
        nn.init.trunc_normal_(self.W, mean=0.0, std=0.3)

        # Normalize embeddings (Replaces tf.clip_by_norm)
        self.embedding.data = F.normalize(self.embedding.data, p=2, dim=1)
        self.L_embedding.data = F.normalize(self.L_embedding.data, p=2, dim=1)
        self.W.data = F.normalize(self.W.data, p=2, dim=1)

    def forward(self, u_i, u_j, this_layer, label):
        # Step 1: Look up embeddings
        u_i_embedding = self.embedding[u_i]
        u_j_embedding = self.embedding[u_j]

        # Step 2: W * u
        u_i_embedding = torch.matmul(u_i_embedding, self.W)
        u_j_embedding = torch.matmul(u_j_embedding, self.W)

        # Step 3: Look up layer embedding
        l_i_embedding = self.L_embedding[this_layer]

        # Step 4: r_i = u_i * W + l
        r_i = u_i_embedding + l_i_embedding
        r_j = u_j_embedding + l_i_embedding

        # Step 6: Compute inner product
        inner_product = torch.sum(r_i * r_j, dim=1)

        # Loss function
        loss = -torch.sum(F.logsigmoid(label * inner_product))

        return loss


# Example usage
if __name__ == "__main__":
    model = MultiNetworkEmb(num_of_nodes=1000, batch_size=32, K=5, node_embedding=128, num_layer=3, layer_embedding=64)

    # Sample input (random indices for batch processing)
    u_i = torch.randint(0, 1000, (32 * (5 + 1),))
    u_j = torch.randint(0, 1000, (32 * (5 + 1),))
    this_layer = torch.randint(0, 4, (32 * (5 + 1),))
    label = torch.rand(32 * (5 + 1))

    loss = model(u_i, u_j, this_layer, label)
    print("Loss:", loss.item())
