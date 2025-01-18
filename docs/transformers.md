---
layout: single
title: Transformers
permalink: /docs/transformers/
toc: true
---

This section describes the attention and transformer.

## Motivation

###  History of Word Embeddings
* **Bag-of-words** was the first technique to create a machine-representation of text. By counting the frequency of words in a piece of text, one could extract its *characteristics*. 
* **TF-IDF** was developed to address the limitations of considering all words equally important, by assigning weights to each word based on its frequency across all documents.  
* **Word2Vec** revolutionized the field, where the models learn to predict the center word given context words (**CBOW**) or the context words given the center word (**Skip-gram**).  
* However, in Word2Vec, the limitation is that since each word is assigned a unique embedding, polysemous words can not be accurately encoded using this technique. 

### Contextualized Word Embeddings

* **ELMo** {% cite peters2018deepcontextualizedwordrepresentations%} originally proposed the concept, contextualized word embeddings help distinguish between multiple meanings of the same word, in case of polysemous words.
* Transformers are able to encode a word using its context owing to their self-attention mechanism, where embedding is updated via weighted combination of the embeddings of all the other words in the text.

## Attention

### Benefits of Attention

* *Information Bottleneck*: Machine Translation was performed mainly using RNN/LSTM blocks of encoder-decoder blocks, where the entire input-vector was represented using a fixed-length encoded vector (encoder output), creating an information bottleneck. With attention, we no longer have to encode input sentences into a single vector. Instead, we let the decoder attend to different words in the input sentence at each step of output generation. This increases the informational capacity, from a single fixed-size vector to the entire sentence (of vectors).  
* *Long-Range Dependencies*: Recurrent models had difficulty dealing with long-range dependencies. Attention addressed this by letting each step of the decoder see the entire input sentence and decide what words to attend to. This cut down path length and made it consistent across all steps in the decoder.  
* *Parallelization*: The process was sequential, as the hidden states were computed word by word, and prevented parallelization. Attention tackled this by reading the entire sentence in one go and computing the representation of each word, based on the sentence, in parallel. 

### Types of Attention

* **Additive Attention** {% cite bahdanau2016neuralmachinetranslationjointly%}: Computed the compatibility function using a single layer feed forward network.  
* **Scaled Dot Product Attention**: Used in Transformers {% cite vaswani2017attention%}, where the scaling factor $$ \frac{1}{d_k} $$ is used .

While both are similar in theoretical complexity, dot-product attention is much faster and more space-efficient and can be implemented using highly optimized matrix multiplications.

While for smaller values for $$d_k$$, the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $$d_k$$. {% cite britz2017massiveexplorationneuralmachine%}. We suspect that for large values of $$d_k$$, The dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 

ToDO: Write specific mehanisms for additive/dot-product attention?
{: .notice--warning}

### Mathematical Representation

Notations are from {% cite weng2020transformer%}, 

| Symbol | Meaning |
|--------|---------|
| $$d$$ | The model size / hidden state dimension / positional encoding size |
| $$h$$ | The number of heads in multi-head attention layer |
| $$L$$ | The segment length of input sequence |
| $$\mathbf{X} \in \mathbb{R}^{L \times d}$$ | The input sequence where each element has been mapped into an embedding vector of shape $$d$$, same as the model size |
| $$\mathbf{W}^k \in \mathbb{R}^{d \times d_k}$$ | The key weight matrix |
| $$\mathbf{W}^q \in \mathbb{R}^{d \times d_k}$$ | The query weight matrix |
| $$\mathbf{W}^v \in \mathbb{R}^{d \times d_v}$$ | The value weight matrix. Often we have $$d_k = d_v = d$$ |
| $$\mathbf{W}^k_i, \mathbf{W}^q_i, \mathbf{W}^v_i \in \mathbb{R}^{d \times d_k/h}$$ | The weight matrices per head |
| $$\mathbf{W}^o \in \mathbb{R}^{d_v \times d}$$ | The output weight matrix |
| $$\mathbf{Q} = \mathbf{X}\mathbf{W}^q \in \mathbb{R}^{L \times d_k}$$ | The query embedding inputs |
| $$\mathbf{K} = \mathbf{X}\mathbf{W}^k \in \mathbb{R}^{L \times d_k}$$ | The key embedding inputs |
| $$\mathbf{V} = \mathbf{X}\mathbf{W}^v \in \mathbb{R}^{L \times d_v}$$ | The value embedding inputs |
| $$S_i$$ | A collection of key positions for the $$i$$-th query $$q_i$$ to attend to |
| $$\mathbf{A} \in \mathbb{R}^{L \times L}$$ | The self-attention matrix between a input sequence of length $$L$$ and itself. $$\mathbf{A} = \text{softmax}(\mathbf{Q}\mathbf{K}^T / \sqrt{d_k})$$ |
| $$a_{ij} \in \mathbf{A}$$ | The scalar attention score between query $$q_i$$ and key $$k_j$$ |
| $$\mathbf{P} \in \mathbb{R}^{L \times d}$$ | Position encoding matrix, where the $$i$$-th row $$p_i$$ is the positional encoding for input $$x_i$$ |


The output of the one attention head is obtained as,

$$ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V $$

For individual query and key vectors $$q_i, k_j \in \mathbb{R}^D$$, the attention score is calculated as:

$$ a_{ij} = softmax(\frac{q_ik_j^T}{\sqrt{d_k}}) = \frac{exp(\frac{q_ik_j^T}{\sqrt{d_k}})}{\sum_{r\in S_i} exp(\frac{q_ik_r^T}{\sqrt{d_k}})} $$


* $$d_v$$ and $$d_k$$ can be different, however attention paper uses the same value 
* $$\mathbf{W}^v$$ is not $$ d \times d$$, but is made by two matrix $$ d \times d_v$$, and $$ d_v \times d$$, by performing low rank factorization {% cite 3b1b-transformers%}

### Transformers Architecture

{: .text-center}
![Transformers Architecture]({{ site.url }}{{ site.baseurl }}/docs/images/transformers-architecture.png)
Fig. Transformers Architecture
{: .text-center}

* Self-attention is used in the encoder modules, and cross-attention is used in the decoder module. 
* The decoder contains two multi-head attention submodules instead of one in each identical repeating module. The first multi-head attention submodule is masked to prevent positions from attending to the future.



#### Encoder and Decoder Stacks
* The decoder can complete partial sequences and extend them as far as you want. It is a transformer with the encoder stack and all its connections surgically removed. Any time you come across a generative/auto-regressive model, such as GPT-X, LLaMA, Copilot, etc., you’re probably seeing the decoder half of a transformer in action.
* Any time you come across an encoder model that generates semantic embeddings, such as BERT, ELMo, etc., you’re likely seeing the encoder half of a transformer in action.

#### Cross-Attention
Cross-attention works just like self-attention with the exception that the key matrix $$K$$ and value matrix $$V$$ are based on the output of the encoder stack (i.e., the final encoder layer), rather than the output of the previous decoder layer. The query matrix $$Q$$ is still calculated from the results of the previous decoder layer. It’s interesting to note that the same embedded source sequence (output from the final layer in the encoder stack) is provided to every layer of the decoder, supporting the notion that successive layers provide redundancy and are all cooperating to perform the same task. 

#### Multiple Attention Heads

* The output $$head_i$$ of each attention head is concatenated to form a single vector, which is passed through a feedforward layer
* Multiple heads of attention can be run in parallel and the results are concatenated in the end

$$\text{MultiHeadAttn}(X_q, X_k, X_v) = [head_1; head_2; \ldots; head_h]W_0$$

#### Masking
To perform masking, we set the attention matrix elements to $$-\infty$$ before taking the softmax so that after softmax it’s 0 (We need to prevent leftward information flow in the decoder to preserve the auto-regressive property).

#### Design Choice FAQ(s)

##### Why Multiple Heads?
 *  Multiple heads lets the model consider multiple words simultaneously. Softmax function amplifies the highest value while squashing the lower ones (resulting in each head tending to focus on a single element).  Multi-head attention provides multiple *representation subspaces* .
  * Bad random initializations of the learnable weights can destabilize the training process. Multiple heads allow the attention mechanism to essentially *hedge its bets*, looking at different transformations or aspects of the hidden features from the previous layer.  

##### Why stacking multiple attention Layers?  
 * They build in redundancy. Now, if any single attention layer messed up, the skip connections and downstream layers can mitigate the issue.  
 * Stacking attention layers also broadens the model’s receptive field. The first attention layer produces context vectors by attending to interactions between pairs of words in the input sentence. Then, the second layer produces context vectors based on pairs of pairs, and so on.  
 * More attention layers resulted in better performance, although the improvement became marginal after 6\. Thanks to skip connections, successive layers don’t provide increasingly sophisticated abstraction as much as they provide redundancy.  


##### Why Residual Connections?  
  * Skip connections {% cite he2015deepresiduallearningimage%} help dampen the impact of poor attention filtering. Even if an input’s attention weight is zero and the input is blocked, skip connections and add a copy of that input to the output. This ensures that even small changes to the input can still have a noticeable impact on the output.   
  * Skip connections preserve the input sentence. There’s no guarantee that a context word will attend to itself in a transformer. Skip connections ensure this by taking the context word vector and adding it to the output. 

##### Why scaling with $$\frac{1}{\sqrt{d_k}} $$  
  * Intuitively, the normalization factor is added to keep the interpretability of the dot product value the same when the dimensionality is changed. Since, otherwise, increasing the dimensions, would increase the dot-product value.   
  * Controls the variance, for dot product of vectors with dimension $$d_k$$, the variance is $$d_k$$.
  * Ensuring efficient and stable training, and keeping the attention distribution interpretable and effective.

#### Time Complexity

The RNN has a complexity of $$O(L \cdot d^2)$$, where:
- $$L$$ represents each time step in sequential processing
- The update equation is $$h_t = tanh(W_h \cdot h_{t-1} + W_x \cdot x_t)$$ with weight matrices of dimensions [d × d] and [d × input_dim]

The Transformer has a complexity of $$O(L^2 \cdot d)$$, with:
- $$L^2$$ complexity from self-attention matrix calculations
- Computing query-key dot products requires $$O(L^2 \cdot d)$$ operations
- Memory requirement of $$O(L^2)$$

The key difference is that RNNs scale linearly with sequence length ($$L$$) but quadratically with hidden dimension ($$d$$), while Transformers scale quadratically with sequence length but linearly with dimension. The quadratic time complexity with respect to the sequence length means that Transformers can be slower for very long sequences. However, their highly parallelizable architecture often results in faster training and inference times on modern GPUs, especially for tasks involving long sequences or large datasets.

## Training Transformers

### Teacher Forcing
The model is fed with the ground truth (True) target sequence at each time step as input, rather than the model’s own predictions. 

* Accelerates training convergence and stabilizes learning: If we do not use teacher forcing, the hidden states of the model will be updated by a sequence of wrong predictions, errors will accumulate, making it difficult for the model to learn.  
* With teacher forcing, when the model is deployed for inference (generating sequences), it typically does not have access to ground truth information and must rely on its own predictions, which can be less accurate.  This discrepancy between training and inference can potentially lead to poor model performance and instability, known as **exposure bias** in literature, which can be mitigated using **scheduled sampling**.


### Scheduled Sampling

* Addresses the discrepancy between the training and inference phases that arises due to teacher forcing, and it helps mitigate the exposure bias generated by teacher forcing.  
* Bridges the *train-test discrepancy* gap between training and inference by gradually transitioning from teacher forcing to using the model’s own predictions during training.  
* Follows a schedule where teacher forcing is dominant in the early stages, and the probability of using the true target as input is reduced (thus increases the probability of using the model’s own predictions) gradually as training progresses.

### Label Smoothing As a Regularizer

* Penalizes the model if it gets overconfident about a particular choice. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.  
* Implemented using the KL divergence loss. Instead of using a one-hot target distribution, we create a distribution that has a reasonably high confidence of the correct word and the rest of the smoothing mass distributed throughout the vocabulary.


## Reading List


| Title                                          |  Topic       |   Comments                                                   |
| --------------------------------------------   | ------------ | ------------------------------------------------------------ |
| [Transformers Primer by Aman.AI](https://aman.ai/primers/ai/transformers/) {% cite Chadha2020DistilledTransformers%}| Transformers | Very comprehensive                                           |
| [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/) {% cite Chadha2020DistilledTransformers%}| Transformers | Great Illustrations 
| [Attention in transformers, visually explained by 3Blue1Brown](https://www.youtube.com/watch?v=eMlx5fFNoYc&ab_channel=3Blue1Brown) {% cite 3b1b-transformers%}| Transformers | Great Visuals and Explanation
| [Some Intuition on Attention and the Transformer by Eugene Yan](https://eugeneyan.com/writing/attention/) {% cite yan2023attention%}| Transformers | Great Visuals and Explanation
| [The Transformer Family by Lilian Weng](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/) {% cite weng2020transformer%}| Advances in Transformers | Advanced transformer post-enhancements 
| [The Transformer Family 2.0 by Lilian Weng](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) {% cite weng2023transformer%}| Advances in Transformers | Update to {% cite weng2020transformer%} the transformer family. Adds a lot of other updates on the transformers, however, some modules (which were not covered in {% cite weng2020transformer%} since it’s very detailed and niche. 

## References


{% bibliography --cited %}
