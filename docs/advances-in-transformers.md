---
layout: single
title: Advances in Transformers
permalink: /docs/advances-in-transformers/
last_modified_at: 2025-01-17T17:00:00-08:00
toc: true
---

## Transformer-XL

The vanilla transformer has limited attention span (within the context window), and the increase of context window increases the computational and memory requirements. No information can flow between the separated fixed-length segments (**context segmentation**) causing issues like 

* Unable to capture long term dependencies  
* Hard to predict the first few tokens in each segment given no or thin context  
* Expensive evaluation, when a segment is shifted to the right by one, the new segment is re-processed from scratch, although there are a lot of overlapped tokens.

Transformer-XL {% cite dai2019transformerxlattentivelanguagemodels%} solves the context segmentation problem by reusing hidden states between segments, and adopting the relative position encoding that is suitable for reused states. More mathematical details are in {%cite weng2020transformer%}.

## Sparse Transformers

The compute and memory cost of the vanilla Transformer grows quadratically with sequence length and thus it is hard to be applied on very long sequences. {% cite child2019generatinglongsequencessparse%} introduced factorized self-attention, through sparse matrix factorization, making it possible to train dense attention networks with hundreds of layers on sequence length up to 16,384, which would be infeasible on modern hardware otherwise.

In vanilla auto-regressive transformers, attention connectivity pattern $$S = \{S_1, S_2, ..., S_n\}$$, i.e., $$S_i$$ records a set of key-positions that the i-th query vector attends to, is defined as $$S_i = \{j; j \leq i\}$$.

In factorized self-attention, $$S_i$$ is decomposed into a tree of dependency, such that for every pair of $$(i, j), j \leq i$$, there is path connecting $$i$$ back to $$j$$, and $$i$$ can attend to $$j$$ either directly or indirectly. Precisely, $$S_i$$ is divided into $$p$$ non-overlapping subset, where $$m^{th}$$ subset is denoted as $$A_i^{(m)} \subset S_i$$.

Therefore the path between the output position and $$i$$ any $$j$$ has a maximum length $$p + 1$$. For example, if $$(j, a, b, c, ..., i)$$ is a path of indices between $$i$$ and $$j$$, we would have $$j \in A_a^{(1)}, a \in A_b^{(2)}, b \in A_c^{(3)}$$, so on and so forth.

{: .text-center}
![Sparse Factorized Attention]({{ site.url }}{{ site.baseurl }}/docs/images/advances-in-transformers-sparse-transformers.png)
Fig. Attention Connectivity Patterns in Transformers (w/ Sparse). Credits: Originally from {% cite child2019generatinglongsequencessparse%}, and annotated from {%cite weng2023transformer%}
{: .text-center}

1. **Strided Attention** with stride ($$l \sim \sqrt{n}$$). This works well with image data as the structure is aligned with strides. In the image case, each pixel would attend to all the previous pixels in the raster scanning order (naturally cover the entire width of the image) and then those pixels attend to others in the same column (defined by another attention connectivity subset).
  $$A_i^{(1)} = \{t, t + 1, ..., i\}, \text{ where } t = \max(0, i - l)$$
  $$A_i^{(2)} = \{j: (i - j) \bmod l = 0\}$$

2. **Fixed Attention** A small set of tokens summarize previous locations and propagate that information to all future locations, where $$c$$ is a hyperparameter                
  $$A_i^{(1)} = \{j: \lfloor\frac{j}{l}\rfloor = \lfloor\frac{i}{l}\rfloor\}$$
  $$A_i^{(2)} = \{j: j \mod l \in \{l-c,...,l-1\}\}$$

There are three ways to use sparse factorized attention patterns in Transformer architecture,

1. One attention type per residual block and then interleave them, $$attention(X) = Attend(X, A^{(n \mod p)})W^0$$ where $$n$$ is current residual block index

2. Set up a single head which attends to locations that all the factorized heads attend to, $$attention(X) = Attend(X, \cup_{m=1}^p A^{(m)})W^0$$

3. Use a multi-head attention mechanism, but different from vanilla Transformer, each head might adopt a pattern presented above, 1 or 2. This option often performs the best.

## Locality Sensitive Hashing (Reformer)
{%cite kitaev2020reformerefficienttransformer%} solves the following issues
* Memory in a model with $$L$$ layers is $$L$$-times larger than in a single-layer model because we need to store activations for back-propagation.
* The intermediate FF layers are often quite large.
* The attention matrix on sequences of length $$L$$ often requires $$O(L^2)$$ in both memory and time.

by proposing
* LSH to reduce $$O(L^2)$$ to $$O(L.logL)$$
* Replace the standard residual blocks with reversible residual layers, which allows storing activations only once during training instead of $$L$$ times (i.e. proportional to the number of layers).

**LSH Attention**

In $$QK^T$$ part of the attention formula, we’re only interested in the largest elements as only large element contribute a lot after softmax. For each query, we are looking for row vectors in Key, closest to the query in high-dimensional space. A hashing scheme $$x \rightarrow  h(x)$$ is **locality-sensitive** if it preserves the distancing information between data points, such that close vectors obtain similar hashes while distant vectors have very different ones. For a fixed random matrix $$R \in R^{d \times b/2}, h(x) = argmax([xR; -xR])$$ . In LSH, a query can only attend to positions in the same hashing bucket, $$S_i = {j : h(q_i) = h(k_j)}$$

1. The attention matrix for full attention is often sparse.
2. Using LSH, we can sort the keys and queries to be aligned according to their hash buckets.
3. Set $$Q=K$$ (precisely $$k_i  = \frac{q_j}{||q_j||}$$
), so that there are equal numbers of keys and queries in one bucket, easier for batching. Interestingly, this *shared-QK* config does not affect the performance of the Transformer.

4. Apply batching where chunks of consecutive queries are grouped together.

{: .text-center}
![LSH Attention]({{ site.url }}{{ site.baseurl }}/docs/images/advances-in-transformers-lsh.png)
Fig. LSH Attention. Credits: From {% cite kitaev2020reformerefficienttransformer%}


## Reversible Residual Network


Using reversible residual layers {% cite gomez2017reversibleresidualnetworkbackpropagation%}, the motivation is to design the architecture in a way that activations at any given layer can be recovered from the activations at the following layer, using only the model parameters. Hence, we can save memory by recomputing the activation during backprop rather than storing all the activations.
Given a layer $$x \rightarrow y$$, the normal residual layer does $$y = x + F(x)$$, but the reversible layer splits both input and output in the pairs $$(x_1, x_2)$$ and $$(y_1, y_2)$$

{: .text-center}
$$y_1 = x_1 + F(x_2), y_2 = x_2 + G(y_1)$$

and reversing can be performed by

{: .text-center}
$$x_1  = y_1- F(x_2), x_2  = y_2- G(y_1)$$

Reformer uses the same idea by using attention layers as $$F$$ and feedforward layers as $$G$$. The memory can be further reduced by chunking the feed-forward computation,

$$Y2 = [Y_2{(1)}, Y_2{(2)}, \cdots, Y_2^{(c)}]$$


## Universal Transformer

Universal Transformer {%cite dehghani2019universaltransformers%} combines self-attention in Transformer with the recurrent mechanism in RNN, aiming to benefit from both a long-term global receptive field of Transformer and learned inductive biases of RNN. Rather than going through a fixed number of layers, Universal Transformer dynamically adjusts the number of steps using **Adaptive Computation Time (ACT)**. If we fix the number of steps, an Universal Transformer is equivalent to a multi-layer Transformer with shared parameters across layers.

Adaptive Computation Time (ACT) is explained visually [here](https://distill.pub/2016/augmented-rnns/#adaptive-computation-time), but in a nutshell, it allows RNN-like architecture to perform different # of computations at each step (instead of one at each step). The number of steps is determined by extra sigmoidal unit (with weights and biases, making it differentiable), outputting a halting probability at immediate step. The process is halted whenever the cumulative probability goes above $$1-\epsilon$$ to ensure the number of steps are discrete and differentiable. 
Given an input sequence of length $$L$$, Universal Transformer iteratively updates the representation $$H_t \in R^{L \times D}$$ at step $$t$$ for an adjustable number of steps. At step 0, $$H^0$$ is initialized to be the same as the input embedding matrix. All the positions are processed in parallel in the multi-head self-attention mechanism and then go through a recurrent transition function.

{: .text-center}
$$A_t = LayerNorm(H^{(t-1)} + MultiHeadAttention(H^{(t-1)} + P^t))$$

$$H_t = LayerNorm(A^{(t-1)} + Transition(A^t))$$

where the transition function is either a separable convolution or a fully-connected neural network that consists of two position-wise affine transformations + one ReLU. $$P^t$$ is positional encoding with additional time dimension. 


## Stabilization for RL (GTrXL)
The Gated Transformer XL {%cite parisotto2019stabilizingtransformersreinforcementlearning%} is one attempt to use transformers for RL using two key changes
1. The layer normalization is only applied on the input stream in a residual module, but NOT on the shortcut stream. A key benefit to this reordering is to allow the original input to flow from the first to last layer.
2. The residual connection is replaced with a GRU-style gating mechanism.

## Sliding Window Attention
{%cite beltagy2020longformerlongdocumenttransformer%}{%cite child2019generatinglongsequencessparse%} Fixed-sized attention block that allows a current token to attend only a specific number of previous tokens (instead of all previous tokens). 

{: .text-center}
![Sliding Window Attention]({{ site.url }}{{ site.baseurl }}/docs/images/advances-in-transformers-sliding-attention.webp)
Fig. Sliding Window Attention. Credits: From {% cite seb-10aipapers2023%}

## Group-Query Attention

{% cite ainslie2023gqatraininggeneralizedmultiquery%} More generalized form of multi-query attention. The motivation behind this is to reduce the number of trainable parameters by sharing the same Keys and Values heads for multiple Query heads, thereby lowering computational requirements.

{: .text-center}
![Sliding Window Attention]({{ site.url }}{{ site.baseurl }}/docs/images/advances-in-transformers-groupquery-attention.png)
Fig. Overview of grouped-query method. Multi-head attention has H query, key, and value heads. Multi-query
attention shares single key and value heads across all query heads. Grouped-query attention instead shares single
key and value heads for each group of query heads, interpolating between multi-head and multi-query attention.. Credits: From {% cite ainslie2023gqatraininggeneralizedmultiquery%}


## References


{% bibliography --cited %}


