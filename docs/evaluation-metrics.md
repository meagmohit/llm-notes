---
layout: single
title: Evaluation Metrics
permalink: /docs/evaluation-metrics/
last_modified_at: 2025-01-17T17:00:00-08:00s
toc: true
---


### Perplexity

Intrinsic evaluation metric to evaluate how well a language model can capture the real word distribution conditioned on the context. A perplexity of a discrete probability distribution  is defined as the exponentiation of the entropy,

$$
2^{H(p)} = 2^{-\sum_x p(x)log_2p(x)}
$$

Given a sentence with $$N$$ words, $$s = (w_1, w_2, \cdots, w_N)$$, assuming that each word has frequency $$1/N$$ the entropy, and perplexity is given by,

$$
H(s) = -\sum_{i=1}^N P(w_i)log_2p(w_i) = -\frac{1}{N} \sum_{i=1}^N log_2p(w_i) \\
$$

$$
2^{H(s)} = (p(w_1), p(w_2), \cdots, p(w_N))^{-1/N}
$$

 A good language model should predict high word probabilities, hence, the smaller perplexity the better. 


## References

{% bibliography --cited %}