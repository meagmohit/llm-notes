---
layout: single
title: Alignment/RLHF
permalink: /docs/alignment-rlhf
toc: true
---

Need to re-iterate, and re-organize
{: .notice--info}

## Motivation

### Hypothesis

Three hypotheses on why RLHF works {%cite yoavg-rl-llm%}

* The diversity hypothesis: during SFT, the model’s output is expected to somewhat match the demonstrated responses. For example, given the prompt “what’s an example of a language?”, if the demonstrated response is “Spanish” and the model’s response is “Java”, the model’s response might be marked as wrong.  
* The negative feedback hypothesis: demonstration only gives the model positive signals (e.g. only showing the model good responses), not negative signals (e.g. showing models what bad responses look like). RL allows us to show models negative signals.  
* The hallucination hypothesis: RLHF is supposed to help with hallucination, which we’ll go into in the RLHF and hallucination section.

## Methods
Comparison data is obtained from a labeling process (via human annotators) that looks like: (prompt, winning_response, losing_response). An example is Anthropic’s HH-RLHF dataset which is publicly available. This is translated into outputting the scores by using a loss function by maximizing the difference in score between the winning response and the losing response (in InstructGPT). 

$$-log(\sigma(r_{\theta}(x, y_w) - r_{\theta}(x, y_l)))$$ where $$y_w$$, $$y_l$$ are winning and losing responses, and $$r_{\theta}$$ is a reward model being trained.

### Architectures for Reward Models

* LM Classifiers: An LLM finetuned as a binary classifiers  
* Value Networks: Regression models to predict a scalar rating  
* Critique Generators: LMs trained to generate an evaluative critique explaining which response is better and why. The critique is used with instruction tuning.

{: .text-center}
![Annotated InstructGPT Performance.]({{ site.url }}{{ site.baseurl }}/docs/images/rlhf-instructgpt-performance.png)
Fig. Annotated InstructGPT Performance. Image Credits {% cite seb-rlhf%}


## Case-Studies

### Instruct-GPT

{%cite ouyang2022traininglanguagemodelsfollow%} 1.3B model, and the human laborers preferred outputs of InstructGPT over GPT-3 (175B). without compromising on the GPT-3 performance. This is believed to be the main approach behind ChatGPT. A limitation of this approach is that it introduces an “alignment tax”: aligning the models only on customer tasks can make their performance worse on some other academic NLP tasks, which can be minimized by “during RL fine-tuning we mix in a small fraction of the original data used to train GPT-3, and train on this data using the normal log likelihood maximization”. Pretraining takes up 98% of the overall compute and data resources.

This reward model generally originates from the LLM created in the prior supervised finetuning step, typically by substituing with a regression layer, which features a single output node.

{: .text-center}
![InstructGPT Methods.]({{ site.url }}{{ site.baseurl }}/docs/images/rlhf-instructgpt-methods.webp)
Fig. Methods for InstructGPT. Image Credits {% cite ouyang2022traininglanguagemodelsfollow%}

{: .text-center}
![Annotated InstructGPT Objective Function]({{ site.url }}{{ site.baseurl }}/docs/images/rlhf-instructgpt-objectivefunction.png)
Fig. Self-Annotated InstructGPT Objective Function

@misc{yoavg-rl-llm,
  author = {Yoav Goldberg},
  year = {2023},
  title = {Reinforcement Learning for Language Models [Blog]},
  url = {https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81},
}

### RLHF in LLAMA2

* Reward Models: two separate reward models are created using human preference data \- one for helpfulness and one for safety. The final reward function that is then used for the model optimization is a linear combination of the two scores.  
* Margin Labels: Unlike the InstructGPT approach that generates multiple outputs and uses a “k choose 2” comparison method, Llama 2’s dataset is based on binary comparisons, and each labeler is presented with only two responses at a time. A margin label is collected alongside binary ranks to indicate the degree of preference (i.e., “significantly better” to “negligibly better”), which can optionally be used in the binary ranking loss via an additional margin parameter to calculate the gap between the two responses.   
* Rejection Sampling: Rejection sampling is used to draw multiple outputs and select the one with the highest reward for the gradient update. PPO is then used to align the model further, making the model’s responses more safe and helpful.   
* Iterative Updates: The Llama-2-chat model evolves through multiple stages (RLHF-v1 to RLHF-v5), with reward models being updated based on the emerging errors from the Llama-2-chat model.

## RLHF & Hallucinations

There are two hypotheses of why LLMs hallucinate {%cite chip-rlhf%}

1. LLMs lack the understanding of the cause and effect of their actions, can be addressed by treating response generation as causal interventions. {%cite ortega2021shakingfoundationsdelusionssequence%} 
2. Hallucination is caused by the mismatch between the LLM’s internal knowledge and the labeler’s internal knowledge. John Schulman, OpenAI co-founder and PPO author, suggested that behavior cloning causes hallucination. During SFT, LLMs are trained to mimic responses written by humans. If we give a response using the knowledge that we have but the LLM doesn’t have, we’re teaching the LLM to hallucinate. \[[John Schulman - Reinforcement Learning from Human Feedback: Progress and Challenges](https://www.youtube.com/watch?v=hhiLw5Q_UFg)\]  
   1. Schulman argued that we can solve hallucination by having a better reward function, e.g. punishing a model more for making things up

From Schulman’s talk, {%cite chip-rlhf%} got the impression that RLHF is supposed to help with hallucination. However, the InstructGPT paper shows that RLHF actually made hallucination worse.

## PPO: Proximal Policy Optimization

PPO {%cite schulman2017proximalpolicyoptimizationalgorithms%} operates on the policy gradient approach, where the agent directly learns a policy, typically parameterized by a neural network. PPO is based on the actor-critic framework, which means it simultaneously trains two components: the actor (the policy) and the critic (the value function). allows PPO to efficiently balance exploration and exploitation by guiding the actor’s policy updates using feedback from the critic. The critic helps compute the advantage function, which quantifies the quality of the actions taken, enabling more informed updates to the policy.

### Clipped Surrogate Loss in PPO

Ratio $$r(\theta)$$ is defined as $$\pi_{\theta}(a|s)$$ / $$\pi_{ref}(a|s)$$, 
where $$\pi_{\theta}$$ and $$\pi_{ref}$$ are current and reference policy, respectively. The loss is then given by,



$$L^{PPO-CLIP}(\theta) = E[min(r(\theta)\hat{A}, clip(r(\theta), \in \hat{A} )]$$

where $$\hat{A}$$ is the advantage function. The clipping mechanism is employed to limit the magnitude of these updates, maintaining stability during training.


Another variant (PPO-penalty), besides the clipped objective, another common approach is to add a KL divergence penalty directly to the objective function. This means the algorithm would penalize the objective based on how much the new policy diverges from the reference policy

$$L^{KL}(\theta) = E[L^{PPO}(\theta) - \beta KL[\pi_{ref} || \pi_{\theta}]$$

### DPO: Direct Preference Optimization

The policy is optimized directly from human preferences (without explicit reward modeling or RL), where it increases the relative log probability of preferred responses to unpreferred ones using a binary cross entropy loss. As compared to PPO

* DPO doesn’t require a direct reward model. Instead, it directly updates the LLM using a classification-like objective.   
* Easier to implement, and computationally more efficient to apply.  
* LLAMA-2 was trained on PPO, while LLAMA-3 used DPO.   
* Zephyr-7B (based on Mistral-7B) used DPO

DPO redefines the RLHF objective by showing that the reward can be rewritten purely as a function of policy probabilities, allowing the LM to implicitly define both the policy and the reward function. This innovation eliminates the need for a separate reward model.

The stability, performance, and computational efficiency of DPO are significant improvements over traditional methods. It eliminates the need for sampling from the LM during fine-tuning, fitting a separate reward model, or extensive hyperparameter tuning.

**DPO’s Binary Cross Entropy Loss**
To compare pairs of model-generated responses (preferred and dispreferred) against human preferences. The model generates two responses for each input, and human annotators indicate which response they prefer. The model then assigns probabilities to each response. The BCE loss function computes the difference between these model-assigned probabilities and the actual human preferences, penalizing the model when it assigns a higher probability to the dispreferred response. By minimizing this loss, DPO adjusts the model’s internal parameters to better align with human preferences.

$$L_{DPO}(\pi_{\theta}; \pi_{ref}) = -E_{(x, y_w, y_l) \sim D} [log \sigma (\beta.log \frac{\pi_{\theta}(y_w | x)}{\pi_{ref}(y_w | x)} - \beta. log \frac{\pi_{\theta}(y_l | x)}{\pi_{ref}(y_l | x)} )]$$

 $$\beta$$ controls how much the model stays close to the reference policy. A larger $$\beta$$ amplifies the contrast between the two responses, making the model more sensitive to preference differences. The High β, the model stays closer to the reference policy, limiting the divergence from the initial policy. This helps retain stability and prevents overfitting to noisy or extreme preferences in the dataset. The Low $$\beta$$, the model is allowed to diverge further from the reference policy, giving it more freedom to optimize for the preferences in the dataset. However, this increases the risk of overfitting or producing less generalizable responses.

Recent models, including Apple Foundational Models {%cite gunter2024appleintelligencefoundationlanguage%} and Tulu3 {%cite lambert2024tulu3pushingfrontiers%} use both PPO and DPO.

### Kahneman-Tversky Optimization (KTO)
Details in {%cite aman-rlhf%} skipped as doesn;t seem important for now

## References


{% bibliography --cited %}
