---
layout: single
title: Emerging Methods
permalink: /docs/emerging-methods/
toc: true
---

## Continued Pre-Training

Goal: To ingest new knowledge in LLMs

{%cite ibrahim2024simplescalablestrategiescontinually%} performed extensive experiments to derive insights into the strategies

* re-warming and re-decaying the learning rate: exact same learning rate schedule that was used during the initial pretraining stage  
* Adding a small portion (e.g., 5%) of the original pretraining data to the new dataset to prevent catastrophic forgetting. Note that smaller fractions like 0.5% and 1% were also effective

{: .text-center}
![continued pretraining]({{ site.url }}{{ site.baseurl }}/docs/images/emerging-methods-continued-pretraining.png)
Fig. A schedule for continued pretraining from {%cite seb-aipapers-2024-part1%}


## Weight Averaging
Weight averaging involves averaging a single model's weights (parameters) at different points in its training process. Typically, it's done towards the end of the training when the model has nearly converged. 

### Stochastic Weight Averaging (SWA)

{%cite izmailov2019averagingweightsleadswider%} where we decay an initially large learning rate, and weights are averaged over several iterations during periods of decayed (but still relatively high) learning rates. Since a model's training trajectory can be uneven, the strategy is to average the models towards the end of the training when the learning rate is low (if a scheduler is used), where the training is nearing convergence. A modified learning rate schedule allows SGD (or other optimizers such as Adam) to continue to bounce around the optimum and explore diverse models instead of simply converging to a single solution.

{: .text-center}
![SWA]({{ site.url }}{{ site.baseurl }}/docs/images/emerging-methods-swa.png)
Fig. Stochastic Weight Averaging (SWA) averages a model's weights towards the end of the training cycle. Image Credits {% cite seb-aipapers-2023%}

Intuition of why SWA works? (From first author of {%cite izmailov2019averagingweightsleadswider%})

Averaging works in cases where the parameters you are averaging are oriented around a local region of low loss, helping to move the solution to a more centred "flat" solution. A flat solution, at a high level, corresponds to parameters that can be perturbed without significantly increasing the loss. These solutions generalize better, in short, because they provide a better compression of the data: they can be represented with fewer bits of precision. It's actually quite intuitive\!

The bigger the model class, the larger these regions of low loss will be, because there are many more parameters that will be consistent with the data, and therefore the greater opportunities to benefit from averaging. Really big models are also hardly trained to completion, so averaging many checkpoints, or across multiple fine tuning runs, can help find more refined solutions more quickly. This is very related to a procedure called SWALP {%cite yang2019swalpstochasticweight%}, which provides training in low precision by combining weights that have been rounded up with those that have been rounded down. Model soups {%cite wortsman2022modelsoupsaveragingweights%} works for exactly the same reason as SWA (and was very inspired by SWA) \- the fine tuning runs each have the same warm start and end up exploring a local region of space.

Theory of linear mode connectivity of model weights {%cite juneja2023linearconnectivityrevealsgeneralization%} suggests that models that start from a similar position, or are fine-tuned in similar ways, end up in the same “region” of loss space, and linearly moving between them can usually get you a similar good (if not better) model, which is similar to SWA. 

### Latest Weight Averaging (LaWA)
 {%cite kaddour2022stopwastingtimesaving%} demonstrated that averaging the weights of the latest k checkpoints, each taken at the end of an epoch, can expedite training progress in terms of loss and accuracy by several epochs. {%cite sanyal2023earlyweightaveragingmeets%} explored a modified version of LaWA with higher learning rates and an earlier start in averaging checkpoints during training. The researchers found that this approach significantly outperformed standard SWA and EMA techniques. 

### WARM 
{%cite ramé2024warmbenefitsweightaveraged%} aims to enhance the RLHF for LLMs. Specifically, the researchers attempt to mitigate reward hacking in LLMs by averaging the weights of finetuned reward models. **Reward hacking** occurs when an LLM learns to manipulate or exploit its reward system's flaws to attain high scores or rewards, without genuinely fulfilling the intended task or achieving the essential objectives. WARM proposes to average the weight of multiple RMs. They use a simple linear average as in stochastic weight averaging. The difference, however, is that the models are not sampled from the same trajectory but are independently created from the pretrained model, as in Model ratatouille. Alternatively, WARM also has a so-called Baklava procedure to sample along a fine-tuning trajectory.

{: .text-center}
![WARM in RLHF]({{ site.url }}{{ site.baseurl }}/docs/images/emerging-methods-warm-rlhf.jpg)
Fig. An outline of how WARM is used in the RLHF process. The only new aspect here is that the method uses a reward model from weight averaging instead of training a single reward modeling (annotated figure from WARM paper. Image Credits {% cite seb-aipapers-2023%}

{: .text-center}
![ratatouille comparison]({{ site.url }}{{ site.baseurl }}/docs/images/emerging-methods-ratatouille-comparison.jpg)
Fig. A comparison between the different model merging and averaging methods. Image Credits {% cite seb-aipapers-2023%}

## Model Merging

Model merging involves combining multiple different trained models into a single model. Model Ratatouille {%cite ramé2023modelratatouillerecyclingdiverse%} proposes to reuse multiple fine-tuned iterations of the identical base model across various diverse auxiliary tasks. 

Model Soups {%cite wortsman2022modelsoupsaveragingweights%} averaging the weights of multiple models finetuned with different hyperparameter configurations often improves accuracy and robustness.” The key difference between this and an ensemble is that no inference time penalty is incurred. 

{: .text-center}
![Model Ratatouille]({{ site.url }}{{ site.baseurl }}/docs/images/emerging-methods-ratatouille.jpg)
Fig. The Model Ratatouille method for model merging. Image Credits {% cite seb-aipapers-2023%}


## Mixture of Experts (MoE)

Mixture of Experts, is a type of ensemble model that combines several smaller *expert* subnetworks. Each subnetwork is responsible for handling different types of tasks or, more concretely, tokens. allocate computational resources more efficiently.

### Switch Transformers
{%cite fedus2022switchtransformersscalingtrillion%}

{: .text-center}
![Switch Transformers]({{ site.url }}{{ site.baseurl }}/docs/images/emerging-methods-switch-transformers.png)
Fig.  Annotated Figure of Switch Transformers. Image Credits {% cite seb-aipapers-2023%}

### Mixtral 8x7B

{%cite jiang2024mixtralexperts%} is a sparse MoE, performing similar to Llama 2 70B. 
Uses 8 experts in-total, and 2 experts per token combining Mistral 7B model. The total number of parameters are 47B (not 56B, since only FFNs are copied) The router reroutes the tokens such that only 13B (<14B) parameters (2x <7B, instead of all <56B) are used at a time for the forward pass, so the training (and especially inference) will be faster compared to the traditional non-MoE approach. 

**Architecture:** Replaces each feed-forward module in a transformer architecture with 8 expert layers. 

{: .text-center}
![Annotated Transformers Architecture]({{ site.url }}{{ site.baseurl }}/docs/images/emerging-methods-mistral.jpg)
Fig. Annotated Transformers Architecture. Image Credits {% cite seb-aipapers-2023%}

Routing module (also known as a gating network or $$G$$) computes the output as $$\sum_{i=1}^8 G(x)_i. E_i(x)$$ where $$E_i$$ are the expert outputs. At first glance, it might seem like Mixtral is simply adding additional parameters to an LLM via these expert (feed-forward) modules to represent a sort of weighted ensemble approach. However, there's an additional tweak: Mixtral is a sparse MoE, which means that only a subset of the experts are used for each input (TopK=2), i.e.,  $$G(x) := Softmax(TopK(x.W_g))$$.

{: .text-center}
![Mixtral of Experts]({{ site.url }}{{ site.baseurl }}/docs/images/emerging-methods-mistral-of-experts.jpg)
Fig. Annotated figure from Mixtral of Experts paper explaining the MoE module. Image Credits {% cite seb-aipapers-2023%}

Expert Specialization: consecutive tokens in text datasets are often assigned to the same experts. Additionally, indentation tokens in Python code are frequently assigned to the same expert

## Proxy Tuning

{%cite liu2024tuninglanguagemodelsproxy%} Proxy-tuning works through a straightforward process at the decoding stage by adjusting the logits of the target LLM. Specifically, it involves calculating the difference in logits between a smaller base model and a finetuned model. This difference is then added to the logits of the target model.

{: .text-center}
![proxy-tuning]({{ site.url }}{{ site.baseurl }}/docs/images/emerging-methods-proxy-tuning.jpg)
Fig. Annotated illustration of proxy-tuning. Image Credits {% cite seb-aipapers-2023%}

**Benefits:**
* It might outperform LoRA in certain contexts,
* It's useful when the large base model is a "black box", and its internal weights are inaccessible.

However, the smaller models must share the same vocabulary as the larger target model.



@misc{liu2024tuninglanguagemodelsproxy,
      title={Tuning Language Models by Proxy}, 
      author={Alisa Liu and Xiaochuang Han and Yizhong Wang and Yulia Tsvetkov and Yejin Choi and Noah A. Smith},
      year={2024},
      eprint={2401.08565},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2401.08565}, 
}

@misc{fedus2022switchtransformersscalingtrillion,
      title={Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity}, 
      author={William Fedus and Barret Zoph and Noam Shazeer},
      year={2022},
      eprint={2101.03961},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2101.03961}, 
}

@misc{jiang2024mixtralexperts,
      title={Mixtral of Experts}, 
      author={Albert Q. Jiang and Alexandre Sablayrolles and Antoine Roux and Arthur Mensch and Blanche Savary and Chris Bamford and Devendra Singh Chaplot and Diego de las Casas and Emma Bou Hanna and Florian Bressand and Gianna Lengyel and Guillaume Bour and Guillaume Lample and Lélio Renard Lavaud and Lucile Saulnier and Marie-Anne Lachaux and Pierre Stock and Sandeep Subramanian and Sophia Yang and Szymon Antoniak and Teven Le Scao and Théophile Gervet and Thibaut Lavril and Thomas Wang and Timothée Lacroix and William El Sayed},
      year={2024},
      eprint={2401.04088},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2401.04088}, 
}


## Reading List


| Title                                          |  Topic       |   Comments                                                   |
| --------------------------------------------   | ------------ | ------------------------------------------------------------ |
| [Model Merging, Mixtures of Experts, and Towards Smaller LLMs](https://magazine.sebastianraschka.com/p/research-papers-in-january-2024) {% cite seb-aipapers-2023%}| MoE/Merging | 


## References

{% bibliography --cited %}