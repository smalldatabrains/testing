# Notes

Attentio is all you need paper review

## Stage 1 : Attention head
https://arxiv.org/abs/1706.03762

Answer the following questions
- What do Q, K and V contains? which question do they answer to?
Regarding the token that we are inspecting
Q : what do i need?
K : what am i?
V : what do i bring?
- Why do we use softmax?
- Why do Q and K come from the same vector?
- Why do we need V?
- Why not simply use the attention matrix as the output? What information should be missing?
- Why do scaling by root(dk) matters?
Summing the dot product with high values of d_k may grow large, thus pushing the value of the softmax to extreme values (either very close to 0 and 1). Scaling by d_k is a good practice.
- Why are different heads not learning the same things? Give some examples of what it's learning
- How to keep the multihead computationnaly efficient?

todo :
- extend to multihead attention
- implement full transformer block
- train on a toy dataset (at character level)
- Describe encoder and decoder parts

## Stage 2 : Diffusion probabilistics models
https://arxiv.org/abs/2006.11239

## Stage 3 : Optimization intuition
https://arxiv.org/abs/1412.6980
