---
layout: single
title: Positional Encodings
permalink: /docs/positional-encodings/
toc: true
---

Crucial component of transformer models, enabling them to understand the order of tokens in a sequence. Needed since self-attention operation is permutation invariant, it is needed to provide order information to the model

## Absolute Positional Encodings

Proposed in {%cite vaswani2017attention%}, conceptually, circular wiggle is added based on the token position and the embedding dimension using sin and cosine functions. Given the token position $$i=1,\cdots,L$$ and the dimension $$\delta=1,\cdots,d$$

$$
\text{PE}(i, \delta) = 
\begin{cases}
\sin(\frac{i}{10000^{2\delta/d}}) & \text{if } \delta = 2\delta' \\
\cos(\frac{i}{10000^{2\delta/d}}) & \text{if } \delta = 2\delta' + 1
\end{cases}
$$


In this way, each dimension of the positional encoding corresponds to a sinusoid of different wavelengths in different dimensions, from $$2$$ to $$10000.2\pi$$

{: .text-center}
![Sinusoidal Positional Encoding]({{ site.url }}{{ site.baseurl }}/docs/images/positional-encoding-sinusoidal.png)
Fig.  Sinusoidal positional encoding with L=32 and d=128. The value is between -1 (black) and 1 (white) and the value 0 is in gray. Credits: From {% cite weng2020transformer%}

**Limitations**: Doesn’t generalize to unseen positions or for sequences longer than the ones encountered during training. This poses a challenge when processing sequences of varying lengths or very long sequences, as the embeddings for out-of-range positions are not learned.


## Learned Positional Encoding

Assigns each element with a learned column vector which encodes its absolute position {%cite gehring2017convolutionalsequencesequencelearning%} and furthermore this encoding can be learned differently per layer {%cite alrfou2018characterlevellanguagemodelingdeeper%}

## Relative Positional Encoding

Proposed in {%cite shaw2018selfattentionrelativepositionrepresentations%}, addresses the limitations of absolute positional encoding by encoding the relative positions between tokens (into $$W^k$$ and $$W^v$$ ) rather than their absolute positions. In this approach, the focus is on the distance between tokens, allowing the model to handle sequences of varying lengths more effectively.

Maximum relative position is clipped to a maximum absolute value of $$k$$ and this clipping operation enables the model to generalize to unseen sequence lengths. Therefore, $$2k+1$$ unique edge labels are considered and let us denote $$P^k$$, $$P^v R^{2k+1}$$ as learnable relative position representations.

$$A^k_{ij} = P^k_{clip(k-1, k)}, A^v_{ij} = P^v_{clip(k-1, k)}, \text{where } clip(x,k) = clip(x, -k, k)$$

**Limitations**: Additional complexity in terms of the computational overhead and memory footprint, mainly for the long sequences. 
Transformer XL {%cite dai2019transformerxlattentivelanguagemodels%} proposed a type of relative positional encoding based on reparametrization of dot-product of keys and queries. Mathematical details are also in {% cite weng2020transformer%}

## Rotary Position Embeddings (RoPE)

Proposed in RoFormer {%cite su2023roformerenhancedtransformerrotary%}, combines benefits of absolute and relative while being parameter efficient. 
Given a token embedding $$x$$ and its position $$pos$$, the RoPE mechanism applies a rotation matrix $$R(pos)$$ to the embedding: $$RoPE(x,pos)=R(pos)⋅x$$

It encodes the absolution position with a rotation matrix and multiplies key and value matrices of every attention layer with it to inject relative positional information at every layer. When encoding relative positional information into the inner product of the $$i^{th}$$ key and the $$j^{th}$$ query, we would like to formulate the function in a way that the inner product is only about the relative position $$(i-j)$$. Rotary Position Embedding (RoPE) makes use of the rotation operation in Euclidean space and frames the relative position embedding as simply rotating feature matrix by an angle proportional to its position index.
A vector $$z$$ in the euclidean space can be rotated counterclockwise by multiplying it with Rz where rotation matrix $$R$$ is defined as,

$$
R = \begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix}
$$

When generalizing to higher dimensional space, RoPE divides the $$d$$-dimensional space into $$d/2$$ subspaces and constructs a rotation matrix $$R$$ of size $$d \times d$$ for token at position $$i$$:

$$
R^{d}_{\Theta, i} = \begin{bmatrix}
\cos i\theta_1 & -\sin i\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin i\theta_1 & \cos i\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos i\theta_2 & -\sin i\theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin i\theta_2 & \cos i\theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos i\theta_{d/2} & -\sin i\theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin i\theta_{d/2} & \cos i\theta_{d/2}
\end{bmatrix}
$$

where in the paper we have,

{: .text-center}
$$\Theta = \theta_i = 10000^{-2(i-1)/d}, \; i \in [1, 2, \dots, d/2]$$

Note that this is essentially equivalent to sinusoidal positional encoding but formulated as a rotation matrix. Then both key and query matrices incorporates the positional information by multiplying with this rotation matrix,


$$q_i^T k_j = \left(R^{d}_{\Theta, i} W^q x_i\right)^T \left(R^{d}_{\Theta, j} W^k x_j\right) = x_i^T W^q R^{d}_{\Theta, j-i} W^k x_j$$, 
{: .text-center}
where $$R^{d}_{\Theta, j-i} = \left(R^{d}_{\Theta, i}\right)^T R^{d}_{\Theta, j}$$

**Limitations**:
* It is specifically designed for certain architectures and may not generalize as well to all transformer variants or other types of models
* mathematical complexity might make it harder to implement and optimize compared to more straightforward positional encoding methods.
* In practice, RoPE might be less effective in transformer models that are designed with very different architectures or in tasks where positional information is not as crucial.

**Benefits**:
* Effectiveness in Long Contexts: RoPE scales effectively with sequence length, making it suitable for LLMs that need to handle long contexts or documents. This is particularly important in tasks like document summarization or question-answering over long passages.


## References


{% bibliography --cited %}





