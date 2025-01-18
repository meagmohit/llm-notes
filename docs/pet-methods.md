---
layout: single
title: Parameter Efficient Tuning (PET) Methods
permalink: /docs/pet-methods/
last_modified_at: 2025-01-17T17:00:00-08:00s
toc: true
---

## LoRA: Low Rank Adaptation
{%cite hu2021loralowrankadaptationlarge%} proposed an efficient training method to finetune or custom-train pre-trained LLMs without excessive memory footprint. LoRA proposes to decompose the weight update matrix $$(\Delta W)$$ into two low-rank matrices $$A$$ and $$B$$, with inner dimension as $$r$$. An additional hyperparameter $$\alpha$$ is used for scaling, $$ W  = (A  \times B) * \alpha /r  $$

{: .text-center}
![LoRA weight updates]({{ site.url }}{{ site.baseurl }}/docs/images/pet-methods-lora.jpg)
Fig. LoRA weight updates. Credits: From {% cite seb-finetuning-llms%}

When using LoRA, we hypothesize that the model requires $$W$$ to be a large matrix with full rank to capture all the knowledge in the pretraining dataset. However, when we finetune an LLM, we don't need to update all the weights and capture the core information for the adaptation in a smaller number of weights. 

Key Takeaways and Important Points from {% cite seb-finetuning-llms%} (Ongoing Research)

* Consistency: Benchmark results are surprisingly consistent across the different runs despite the inherent randomness of LLM training or when training models on GPUs.  
* QLoRA {%cite dettmers2023qloraefficientfinetuningquantized%} quantized LoRA, quantizes the pre-training weights to 4-bits, to further reduce the memory footprint. {% cite seb-finetuning-llms%} found that QLoRA saves GPU memory however at the cost of increased training runtime (caused by the additional quantization and dequantization of the pretrained model weights in QLoRA). Further, with QLoRA the model performance was similar to LoRA.   
* Cosine annealing scheduler to the LoRA finetuning improved the SGD performance noticeably. However, it has less impact on Adam and AdamW optimizers and makes almost no difference.  
* SGD vs Adam Optimizers: Adam optimizers maintain two moving averages for each model parameter: the first moment (mean) of the gradients and the second moment (uncentered variance) of the gradients, i.e., Adam optimizers store two additional values for each single model parameter in memory. If we are working with a 7B parameter model, that's an extra 14B parameters to track during training. SGD optimizers don't need to track any additional parameters during training, so a question is: what advantage does swapping Adam by SGD have on the peak memory requirements when training LLMs. Swapping Adam optimizers with SGD may not be worthwhile when LoRA's r is small. However, it may be worthwhile when we are increasing r.  
* Multi-epoch training might not benefit instruction finetuning since it can deteriorate the results. This performance decline is likely due to increased overfitting, which warrants additional investigation.  
* Enabling LoRA for more layers, in addition to the Query and Value matrices shows improved performance.  
* A common rule of thumb for choosing alpha is two times $$r$$.   
* Data quality is very important for finetuning. According to the LIMA {%cite zhou2023limaalignment%} paper, a 65B Llama model finetuned on LIMA noticeably outperforms a 65B Llama model finetuned on Alpaca.  
* Choosing the best rank: choosing an r that is too large could result in more overfitting. On the other hand, a small r may not be able to capture diverse tasks in a dataset. In other words, the more diverse the tasks in the dataset, the larger the r should be.

Apple Intelligence Foundation Language Models {%cite gunter2024appleintelligencefoundationlanguage%} use LoRA for on-device task specialization of LLMs.

{%cite biderman2024loralearnsforgets%} experimentally demonstrates that LoRA learns less and forgets less as compared to full finetuning. Hence, full finetuning is better for absorbing new knowledge from more distant domains but leads to more forgetting of previously learned tasks. LoRA, by changing fewer parameters, learns less new information but retains more of the original capabilities. More details in {%cite seb-aipapers-2024-part1%}.


## QLoRA: Quantized LoRA
In {%cite dettmers2023qloraefficientfinetuningquantized%}, the low-rank matrices are quantized, meaning their numerical precision is reduced. This is done by mapping the continuous range of values in these matrices to a limited set of discrete levels. This process reduces the model's memory footprint and computational demands, as operations on lower-precision numbers are less memory-intensive. 

## DoRA: Weight-Decomposed Low-Rank Adaptation 
{%cite liu2024doraweightdecomposedlowrankadaptation%} extended LoRA by first decomposing a pretrained weight matrix into two parts: a magnitude vector $$m$$ and a directional matrix $$V$$. This decomposition is rooted in the idea that any vector can be represented by its length (magnitude) and direction (orientation), and here we apply it to each column vector of a weight matrix. Once we have $$m$$ and $$V$$, DoRA applies LoRA-style low-rank updates only to the directional matrix $$V$$, while allowing the magnitude vector $$m$$ to be trained separately.

{: .text-center}
![DoRA]({{ site.url }}{{ site.baseurl }}/docs/images/pet-methods-dora.png)
Fig. Annotated Illustration of DoRA from {%cite seb-aipapers-2024-part1%}

A two-step approach gives DoRA more flexibility than standard LoRA. Rather than uniformly scaling both magnitude and direction as LoRA tends to do, DoRA can make subtle directional adjustments without necessarily increasing the magnitude. The result is improved performance and robustness, as DoRA can outperform LoRA even when using fewer parameters and is less sensitive to the choice of rank.



<!-- ## Reading List


| Title                                          |  Topic       |   Comments                                                   |
| --------------------------------------------   | ------------ | ------------------------------------------------------------ |
| [Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) {% cite seb-finetuning-llms%}| PET Methods| Good overview of LoRA and practical tips for using it
| [Noteworthy AI Research Papers of 2024 (Part One)](https://magazine.sebastianraschka.com/p/ai-research-papers-2024-part-1) {% cite seb-aipapers-2024-part1%}| PET Methods| 6 Research Papers of 2024-H1
| [Improving LoRA: Implementing Weight-Decomposed Low-Rank Adaptation (DoRA) from Scratch](https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch) {% cite seb-improving-lora%}| PET Methods| DoRA overview in-depth
| [Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments](https://lightning.ai/pages/community/lora-insights/) {% cite seb-finetuning-llm-lightningAI%}| PET Methods| Deep-dive of {% cite seb-finetuning-llms%} -->


## References


{% bibliography --cited %}



