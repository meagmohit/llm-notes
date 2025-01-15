---
layout: single
title: Scaling Laws
permalink: /docs/scaling-laws
toc: true
---

Need to re-iterate
{: .notice--info}

## Core/Basics

### History
Researchers like Leslie Valiant and Michael Kearns explored the relationship between model size and performance through the framework of probably approximately correct (PAC) learning and the VC (Vapnik-Chervonenkis) dimension {%cite valiant1984theory%} of hypothesis classes. Their work showed that the sample complexity of a learning algorithm—that is, the number of examples required to generalize well—scales with the VC dimension of the hypothesis class, which quantifies the model’s capacity or complexity. 

### Motivation
Hyperparameter tuning has a huge cost (especially on large models). Scaling laws provide guidance to perform tuning on the smaller models, and apply the insights on the larger models. 

### Basics of Power Laws
Zipf’s law in NLP: frequency of a word in a text is inversely proportional to the rank of the word in its table. In other words, the most frequent word occurs approximately twice as often as the second most frequent word, three times as often as the third most frequent word, and so on.

## Seminal Works

### OpenAI Scaling Law

OpenAI researchers {%cite kaplan2020scalinglawsneurallanguage%} demonstrated that, for large deep learning models, increasing model size, dataset size, and compute resources consistently reduces model loss in a power-law relationship.  This work laid the groundwork for quantifying the performance gains achievable through scaling, showing that, under specific conditions, larger models trained with more data and compute can achieve significantly higher accuracy.

$$L(N, D) \propto N^{-α} + D^{-β}$$

* L is the model loss  
* N is the number of model parameters  
* D is the dataset size (or \# of tokens in the training set)  
* α and β are scaling exponents that determine the impact of model size and dataset size on model loss

{: .text-center}
![Scaling Laws OpenAI]({{ site.url }}{{ site.baseurl }}/docs/images/scaling-laws-openai.png)
Fig. Language modeling performance improves smoothly as we increase the model size, datasetset
size, and amount of compute used for training. For optimal performance all three factors must be scaled
up in tandem. Empirical performance has a power-law relationship with each individual factor when not
bottlenecked by the other two. {%cite kaplan2020scalinglawsneurallanguage%} 

### Chinchilla Scaling Law

The Chinchilla findings {%cite hoffmann2022trainingcomputeoptimallargelanguage%} revealed that, particularly in compute-constrained settings, increasing data often yields greater benefits than increasing model size alone, leading to improved performance and cost-efficiency. In experiments, a smaller model (Chinchilla with 70 billion parameters) trained on a larger dataset outperformed a much larger model with 175 billion parameters trained on less data. 

$$L(N, D) = 406.4N^{-0.34} +  410.7^{-0.28} + 1.69$$

## Key Points/FAQ(s)

* Hold on many different kind of phenomenon  
  * Dataset Size, Compute and Number of parameters  
    * Loss and Dataset size is linear on a log-log plot  
  * Hold in many domains including machine translation, speech recognition, language modeling, object recognition, etc. 	  
* Conceptual Foundations  
  * Why is it power law or linear in log-log?  
    * Toy task of mean of gaussian distributed RVs, error is σ²/N , i.e., log(Error) is \-log(N) which is a scaling law.  
    * All classical regression models have 1/N scaling.   
    * What do exponents mean? It represents the hidden dimension (intrinsic dimensionality) of the data.  
      * log(error) \= (-1/d) log(N)  
* Double Descent Phenomenon  
  * Performance improves, and then gets worse, and then improves again with increased model size, data and compute.   
* Scaling Hypothesis  
  * The strong scaling hypothesis is that, once we find a scalable architecture like self-attention or convolutions, which like the brain can be applied fairly uniformly, we can simply train ever larger NNs and ever more sophisticated behavior will emerge naturally as the easiest way to optimize for all the tasks & data. More powerful NNs are ‘just’ scaled-up weak NNs, in much the same way that human brains look much like scaled-up primate brains.  
  * AI critics often say that the long tail of scenarios for tasks like self-driving cars or natural language can only be solved by true generalization & reasoning; it follows then that if models solve the long tail, they must learn to generalize & reason  
* The Bitter Lesson   
  * Throw scale, compute and data to the hardest problems to solve. Examples are {%cite chen2020generative %}, {%cite polu2020generative %}

### Why do big models work?
Big models work because they encode a dizzyingly vast number of sub-models in an extremely high-dimensional abstract space, representing countless small sub-models (Orseau et al 2020) interpolating over data, one of which is likely to solve the problem well, and so ensures the problem is soluble by the overall model. They function as an ensemble: even though there countless overfit sub-models inside the single big model, they all average out, leading to a preference for simple solutions. This Occam’s razor biases the model towards simple solutions which are flexible enough to gradually expand in complexity to match the data.

### Practical Considerations

* **Diminishing Returns:** As model size and dataset size increase, the improvements in performance tend to slow down. This phenomenon, known as diminishing returns, suggests that doubling the model size may not always double the performance gain.  
* **Data Quality:** Scaling is not purely about size—data quality plays a crucial role, particularly at larger scales. Datasets need to be diverse and high-quality to maximize scaling benefits, as noisy or low-quality data can lead to suboptimal outcomes


## Case Study: GPT-3 Scaling

{: .text-center}
![Scaling GPT3]({{ site.url }}{{ site.baseurl }}/docs/images/scaling-laws-gpt3.jpg)
Fig. GPT-3: not even that much compute—3640 petaflop/s-day, only 2× their estimate for AlphaGo Zero, 1860. Credits: {%cite gwern-scaling%} 

GPT-3 represents ~103 on this chart, leaving plenty of room for further loss decreases—especially given the uncertainty in extrapolation:

{: .text-center}
![OpenAI Contradiction]({{ site.url }}{{ site.baseurl }}/docs/images/scaling-laws-openai-contradiction.png)
Fig. Far beyond the model sizes we study empirically, we find a contradiction between our equations
for $$L(C_{min})$$ and $$L(D)$$ due to the slow growth of data needed for compute-efficient training. The intersection
marks the point before which we expect our predictions to break down. The location of this point is highly
sensitive to the precise exponents from our power-law fits. Credits: {%cite kaplan2020scalinglawsneurallanguage%} 

GPT-3 continues to scale as predicted. {%cite brown2020languagemodelsfewshotlearners%} 
 

{: .text-center}
![Smooth Scaling of Performance/Compute]({{ site.url }}{{ site.baseurl }}/docs/images/scaling-laws-performance-compute.png)
Fig. Smooth scaling of performance with compute. Performance (measured in terms of cross-entropy
validation loss) follows a power-law trend with the amount of compute used for training. The power-law behavior
observed in [KMH+20] continues for an additional two orders of magnitude with only small deviations from the
predicted curve. For this figure, we exclude embedding parameters from compute and parameter counts. Credits: {%cite brown2020languagemodelsfewshotlearners%} 



## Reading List


| Title                                          |  Topic       |   Comments                                                   |
| --------------------------------------------   | ------------ | ------------------------------------------------------------ |
| [The Scaling Hypothesis](https://gwern.net/scaling-hypothesis) {% cite gwern-scaling%}| Scaling Laws | Great discussion and overview, and thought provoking (Long Read)
| [Scaling Laws in Large Language Models](https://hackernoon.com/scaling-laws-in-large-language-models) {% cite hackernoon-scaling%}| Scaling Laws | Great Quick Overview

## References


{% bibliography --cited %}

