import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim

# Model Visualization
from torchviz import make_dot



# Attention is all you need implementation


# Parameters
TEXT = "hello world, attention is all you need! " * 20 # len(TEXT) = 40 * 20 = 800
T = 10
EPOCHS = 20
LR = 0.0001
BATCH = 1
chars = sorted(list(set(TEXT)))
vocab_size = len(chars) # contains characters and marks
print("There are ", vocab_size, "characeters in the dataset")
print(chars)

# Vocabulary naive embedding with dimension nb_char using dictionnary comprehension
id2char = {i:c for i,c in enumerate(chars)}
char2id = {c:i for i,c in enumerate(chars)}

print(char2id)
print(id2char)


# Dataset construction : T chars --> T +1 chars
def make_dataset(X, T):
    # X and X shifted one step + padding
    X_encoded = torch.tensor([char2id[letter] for letter in X]) # encoding (len(X))
    print(X_encoded[0:11]) # include on the left, exclude on the right
    X=[]
    Y=[]
    for i in range(len(X_encoded)-T): 
        X.append(X_encoded[i:i+T]) # (len(X)-T, T), we can set N = len(X)-T
        Y.append(X_encoded[i+T].item()) # (len(X)-T, 1) -> (len(X) - T) because .item() removes the tensor dimension
        # we must return tensors
    print(X[0])
    print(Y[0])
    X_tensor = torch.stack(X)
    Y_tensor = torch.tensor(Y)
    print("X shape : ", X_tensor.shape)
    print("Y shape : ", Y_tensor.shape)
    return X_tensor, torch.tensor(Y, dtype=torch.long) # return (N,T) and (N,1) as input / output

# Important note:
# in the paper input is (N,T) but output is also (N,T) : they say input shifted right.

X_data, Y_data = make_dataset(X=TEXT, T=T)

# Single head attention
class SingleHeadAttention(nn.Module):
    def __init__(self, vocab_size=17, D = 12 , d_k = 32, d_v=32, T = 10):
        super().__init__()
        # B Batch : qty of sentences that we batch at once (here only 1 for the exemple), size it up to availble memory
        # T : sentence dimension, tokens qty (we can pad for smaller sentence)
        # D Embedding dimension (feature per token)
        
        # Q, K and V dimensions
        d_k = d_k
        d_v = d_v
        
        # Learnable layer with weights
        self.embedding = nn.Embedding(vocab_size,D) # (vocab_size, D)
        self.Wq = nn.Linear(D, d_k) # this is a function X --> XW + b
        self.Wk = nn.Linear(D, d_k)
        self.Wv = nn.Linear(D, d_v)
        self.fc_out = nn.Linear(d_v*T, vocab_size)

        self.scale = d_k ** 0.5

        # Activation function
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        # X is of size (N, T)
        X = self.embedding(X) # (N, T, D) because embedding is a lookup table that transform id (1 dimension to D dimensions)
        Q = self.Wq(X) # (N, T, d_k)
        K = self.Wk(X) # (N, T, d_k)
        V = self.Wv(X) # (N, T, d_v)
        Kt = torch.transpose(K, -2, -1) # swap last 2 dimentions (N, d_k, T)
        scores = torch.matmul(Q, Kt) / self.scale # (N, T, d_k) * (N, d_k, T) --> (N, T, T)
        # attention = scores * 2
        attention = self.softmax(scores) # (N, T, T)
        output = torch.matmul(attention, V) # (N, T, T) * (N, T, d_v) --> (N, T, d_v)
        flat = torch.flatten(output, start_dim=1)

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, NH=6):
        super().__init__()

        # nn.Sequential if fix size, nn.ModuleList if modulable qty of layer
        self.heads = nn.ModuleList(
            [SingleHeadAttention() for i in range(NH)]
        )

        self.Wo = nn.Linear()

    def forward(self, X):
        X = torch.cat([])
        output = self.Wo(X)

        return output

if __name__ == "__main__":
    input = "I like cats very much"
    model = MultiHeadAttention()

    print("Model Layers : ", model)
    # Png model architecture
    dummy_input = torch.randint(0, vocab_size, (1, T))  # input fictif (B, T)
    attention, yhat = model(dummy_input)  # yhat = logits
    make_dot(yhat, params=dict(list(model.named_parameters()))).render("Transformers", format="png")

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum = 0.9)
    loss_fn = nn.CrossEntropyLoss()

    # exemple
    X = X_data
    Y = Y_data


    attention, logits = model(X)

    print("Input Shape : ", X.shape)
    print("Attention Shape : ", attention.shape) # each token build his own attention matrix (T,T)
    print(attention)
    print("Logits Shape : ", logits.shape)
    print(logits)

    # Sanity check
    summation = attention.sum(dim=-1)
    print("Attention Rows sum to ", summation[0])

    # training loop for a next character model prediction
    for epoch in range(100):
        total_loss=0.0
        for input, target in zip(X_data, Y_data):
            input = input.unsqueeze(0)
            target = target.unsqueeze(0)
            optimizer.zero_grad()
            attention, logits = model(input)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Visualization
        plt.clf() # clear previous figure
        plt.imshow(attention[0].detach().numpy(), cmap = "Blues", vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig("./figures/attention_weights_epoch_"+str(epoch +1)+".png", dpi=150, bbox_inches="tight")
        print(f"Epoch {epoch + 1} | loss = {total_loss/len(X_data)}")