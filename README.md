# Notes

Attention is all you need paper review

## Stage 1 : Attention head
https://arxiv.org/abs/1706.03762

Answer the following questions
- What do Q, K and V contains? which question do they answer to?
Regarding the token that we are inspecting
Q : what information am I looking for?
K : what information do I advertise?
V : what information do I transmit if selected?
- Why do we use softmax?
turn matching score into probabilities (0 to 1)
- Why do Q and K come from the same vector? (matching)

- Why do we need V? (communication)
- Why not simply use the attention matrix as the output? What information should be missing?
- Why do scaling by root(dk) matters?
Summing the dot product with high values of d_k may grow large, thus pushing the value of the softmax to extreme values (either very close to 0 and 1). Scaling by d_k is a good practice.
- Why are different heads not learning the same things? Give some examples of what it's learning
Every head is initialized randomly and learn different representation during training because the parameters are independent : semantic, syntax, phrase order, subjects, tenses, characters relationship.
- How to keep the multihead computationnaly efficient?
We scale d_k, d_v and d_q down with the quantity of heads that we are launching in paralell.

Transformer are different than attention. They are attention + stability engineering (residual connections, layernorm, FFN)

Embedding and fc_out are shared between head so it should be in the MultiHeadAttention class (not in the SingleHeadAttention class)

todo :
- experiments on training instability : remove residuals (training divergence), layernorm (instability in loss curves) or decrease number of heads (performance drop)
- implement full transformer block (encoder + decoder)
- train on a toy dataset (at character level)
- Describe encoder and decoder parts


## Stage 2 : Diffusion probabilistics models
https://arxiv.org/abs/2006.11239

## Stage 3 : Optimization intuition
https://arxiv.org/abs/1412.6980
