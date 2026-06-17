import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim


# Attention is all you need implementation


# Parameters
TEXT = "hello world, attention is all you need! " * 20
T = 16
EPOCHS = 20
LR = 0.01
chars = sorted(list(set(TEXT)))
vocab_size = len(chars)
print("There are ", vocab_size, "in the dataset")

# Vocabulary naive embedding with dimension nb_char using dictionnary comprehension
char2id = {i:c for i,c in enumerate(chars)}
id2char = {c:i for i,c in enumerate(chars)}

print(char2id)
print(id2char)


# Dataset construction
def make_dataset(X):

    
    # X and X shifted one step + padding
    dataset = None
    # must return a tensor
    return dataset


# Single head attention
class SingleHeadAttention(nn.Module):
    def __init__(self, D = 12 , d_k = 32, d_v=32):
        super().__init__()
        # B Batch : qty of sentences that we batch at once (here only 1 for the exemple), size it up to availble memory
        # T : sentence dimension, tokens qty (we can pad for smaller sentence)
        # D Embedding dimension (fetaure per token)
        
        # Q, K and V dimensions
        d_k = d_k
        d_v = d_v
        
        self.Wq = nn.Linear(D, d_k) # this is a function X --> XW + b
        self.Wk = nn.Linear(D, d_k)
        self.Wv = nn.Linear(D, d_v)

        self.scale = d_k ** 0.5

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)
        Kt = torch.transpose(K, -2, -1) # swap last 2 dimentions
        scores = torch.matmul(Q, Kt) / self.scale
        # attention = scores * 2
        attention = self.softmax(scores)
        output = torch.matmul(attention, V)

        return attention, output


if __name__ == "__main__":
    input = "I like cats very much"
    model = SingleHeadAttention()
    optimizer = optim.SGD(model.parameters, lr=LR, momentum = 0.9)
    loss_fn = nn.CrossEntropyLoss()

    # exemple
    X = torch.sin(torch.linspace(0, 10, 100)).unsqueeze(-1).repeat(1, 12)
    X = X.unsqueeze(0)

    # X = X.unsqueeze(0) # 1 batch dimension

    attention, output = model(X)

    print("Input Shape : ", X.shape)
    print("Attention Shape : ", attention.shape) # each token build his own attention matrix (T,T)
    print(attention)
    print("Output Shape : ", output.shape)
    print(output)

    # Sanity check
    summation = attention.sum(dim=-1)
    print("Attention Rows sum to ", summation[0])

    # Visualization
    plt.imshow(attention[0].detach().numpy(), cmap = "Blues", vmin=0, vmax=1)
    plt.colorbar()
    plt.savefig("attention_weights_softmax.png", dpi=150, bbox_inches="tight")

    # config
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum = 0.9)
    
    # training loop for a next character model prediction
    for epoch in range(10):

        for input, target in dataset:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()