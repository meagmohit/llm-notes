---
layout: single
title: Large Language Models (LLMs)
permalink: /docs/large-language-models/
last_modified_at: 2025-01-17T17:00:00-08:00s
toc: true
---


## Models 

### CoVE: Contextual Word Vectors (Before Transformers)

{%cite mccann2018learnedtranslationcontextualizedword%} proposed CoVE, a word embedding (function of the entire input sequence) learned by an encoder in an attentional seq-to-seq machine translation model.

{: .text-center}
![The NMT base model used in CoVe]({{ site.url }}{{ site.baseurl }}/docs/images/llm-cove.png)
Fig.  The NMT base model used in CoVe. Credits: From {% cite weng2019LM%}

### ELMo: Embeddings from Language Model

Original Paper: {%cite peters2018deepcontextualizedwordrepresentations%}, Related Papers: {%cite mccann2018learnedtranslationcontextualizedword%}
{%cite peters2017semisupervisedsequencetaggingbidirectional%}

Instead of using a fixed embedding for each word, the embeddings for the words were generated based on the context (the full sentence). ELMo used a stack of bi-directional LSTM layers, pre-trained on the language modeling task (i.e., predicting the next word). 

{: .text-center}
![ELMo Model]({{ site.url }}{{ site.baseurl }}/docs/images/llm-elmo.png)
Fig.  The contextualized word embeddings were generated by concatenating the hidden layers (from both forward and backward model), performing weighted sum over the layers based on given task. 

**Downstream Tasks**
* Semantic task: The word sense disambiguation (WSD) task emphasizes the meaning of a word given a context. The biLM top layer is better at this task than the first layer.
* Syntax task: The part-of-speech (POS) tagging task aims to infer the grammatical role of a word in one sentence. A higher accuracy can be achieved by using the biLM first layer than the top layer.

The comparison study indicates that syntactic information is better represented at lower layers while semantic information is captured by higher layers.

### Cross-View Training

In ELMo the unsupervised pre-training and task-specific learning happen for two independent models in two separate training stages. CVT {%cite clark2018semisupervisedsequencemodelingcrossview%}combines them into one unified semi-supervised learning procedure where the representation of a biLSTM encoder is improved by both supervised learning with labeled data and unsupervised learning with unlabeled data on auxiliary tasks. 

More details are in {% cite weng2019LM%} but skipped since it’s very niche.

### Generative Pre-Training (GPT) aka OpenAI Transformer
Original Paper: {%cite radford2018improving%}

* Pre-trained with language modeling tasks, using only the decoder layers (stacked 12 decoder layers)
* Used clever input transformations for different downstream tasks
* Performed supervised finetuning the base model for all downstream tasks (added the LM loss as an auxiliary loss to accelerate convergence during training, and to improve the generalization of the supervised model). 

{: .text-center}
![GPT]({{ site.url }}{{ site.baseurl }}/docs/images/llm-gpt.png)
Fig.  (left) Transformer architecture and training objectives used in this work. (right) Input
transformations for fine-tuning on different tasks. We convert all structured inputs into token
sequences to be processed by our pre-trained model, followed by a linear+softmax layer. Credits: {%cite radford2018improving%}

$$d_{model} = 768$$ and feed-forward layer dimension is 3072, 117million parameters.
**Limitation**: Unidirectional in nature, can only be used to predict the future left-to-right context

### ULMFit
Original Paper: {%cite howard2018universallanguagemodelfinetuning%}
ULM-FiT first introduced the generative pre-trained language model and task-specific fine tuning. Proposed three key ideas to achieve good transfer learning results
1. General LM pre-training: on larger wiki texts
2. Target task LM-fine tuning: Proposed two training techniques
	* Discriminative fine-tuning: Motivated by the fact that different layers of LM capture different types of information. Each layer was tuned with different learning rates, nl  for the nth layer, where n is the learning rate for the first layer.
	* Slanted triangular learning rates (STLR):  Special learning rate scheduling that first linearly increases the learning rate and then linearly decays it. The increase stage is short so that the model can converge to a parameter space suitable for the task fast, while the decay period is long allowing for better fine-tuning
3. Target task classifier fine-tuning: The pretrained LM is augmented with two standard feed-forward layers and a softmax normalization at the end to predict a target label distribution.
    * Concat pooling extracts max-polling and mean-pooling over the history of hidden states and concatenates them with the final hidden state.
	* Gradual unfreezing helps to avoid catastrophic forgetting by gradually unfreezing the model layers starting from the last one. First the last layer is unfrozen and fine-tuned for one epoch. Then the next lower layer is unfrozen. This process is repeated until all the layers are tuned.

{: .text-center}
![Three training stages of ULMFiT]({{ site.url }}{{ site.baseurl }}/docs/images/llm-ulmfit.png)
Fig. Three training stages of ULMFiT. Credits: From {% cite weng2019LM%}

### BERT
Original Paper: {%cite devlin2019bertpretrainingdeepbidirectional%}

**Key Points**
* Direct descendant of GPT, and compared to GPT main improvement is bi-directional. 
* First token was provided `[CLS]`` token, stands for classification (as in the output sequence, the first output token embedding was used for classification tasks)
* To make BERT better at handling relationships between multiple sentences, the pre-training process includes an additional task: Given two sentences (A and B), is B likely to be the sentence that follows A, or not?)
* Per-trained BERT can be used for creating contextualized word embeddings. The paper examines 6 different choices (which layers to choose?) specific to the downstream tasks. 

**Downstream Tasks**
* Masked Language Modeling
	* Randomly masks 15% of tokens in each sentence and asks the model to predict the missing word. Because if we only replace masked tokens with a special placeholder `[MASK]``, the special token would never be encountered during fine-tuning. Hence, BERT employed several heuristic tricks:
		* with 80% probability, replace the chosen words with `[MASK]`;
		* with 10% probability, replace with a random word;
		* with 10% probability, keep it the same.
	* The model only predicts the missing words, but it has no information on which words have been replaced or which words should be predicted. The output size is only 15% of the input size.
* Next Sentence Prediction
	* auxiliary task on training a binary classifier for telling whether one sentence is the next sentence of the other

{: .text-center}
![BERT input representation.]({{ site.url }}{{ site.baseurl }}/docs/images/llm-bert-input.png)
Fig. BERT input representation. Credits: From {% cite devlin2019bertpretrainingdeepbidirectional%}

**Embeddings in BERT**
* **WordPiece tokenization embeddings**: The WordPiece model was originally proposed for Japanese or Korean segmentation problems. Instead of using naturally split English word, they can be further divided into smaller sub-word units so that it is more effective to handle rare or unknown words.
* **Segment embeddings**: If the input contains two sentences, they have sentence A embeddings and sentence B embeddings respectively and they are separated by a special character `[SEP]`; Only sentence A embeddings are used if the input only contains one sentence.
* **Position embeddings**: Positional embeddings are learned rather than hard-coded.

**Model Types**
* **BERT Base**: 12 encoder layers, 768 hidden units (in feedforward), and 12 attention heads.
* **BERT Large**: 24 encoder layers, 1024 hidden units (in feedforward), and 24 attention heads.

{: .text-center}
![BERT downstream tasks.]({{ site.url }}{{ site.baseurl }}/docs/images/llm-bert-tasks.png)
Fig. Illustrations of Fine-tuning BERT on Different Task. Credits: From {% cite devlin2019bertpretrainingdeepbidirectional%}

### ALBERT: A Lite BERT

Proposed by {%cite lan2020albertlitebertselfsupervised%}, is a lightweight version of BERT, 1.7x faster in training with 18x fewer parameters.
* Factorized Embedding Parameterization: WordPiece tokenization embedding size E is configured to be the same as the hidden state size H. If we want to increase the model size ($$H$$), we need to increase the vocabulary size (V) as well. Using factorized embedding parameterization, the large vocabulary embedding matrix of size $$V \times H$$ is decomposed into $$V \times E$$ and $$E \times H$$. Given $$H >> E$$, factorization can result in greater parameter reduction.
* Cross-layer Parameter Sharing
* Sentence-Order Prediction (SOP): Replaces next sentence prediction (NSP) task of BERT by using (a) Positive sample: two consecutive segments from the same document, and (b) Negative sample: same as above, but the segment order is switched.

### GPT-2
Original Paper: {%cite radford2019language%}
* Direct suppressor of GPT, 1.5B parameters (10x more than the original GPT), 48 transformer layers, and it achieves SOTA results on 7 out of 8 tested language modeling datasets in a zero-shot transfer setting without any task-specific fine-tuning.
* The pre-training dataset contains 8 million Web pages collected by crawling qualified outbound links from Reddit.
* The pre-training task for GPT-2 is solely language modeling
* Tokenization: Uses BPE on UTF-8 Byte Sequences (4 Bytes for 1 character). GPT-2 prevents BPE from merging characters across categories (thus dog would not be merged with punctuations like., ! and ?). These tricks help increase the quality of the final byte segmentation.
* Involved only a few architectural changes - (a) layer normalization was moved to the input of each sub-block, similar to a residual unit of type “building block”, etc.

### RoBERTa: Robustly optimized BERT
{%cite liu2019robertarobustlyoptimizedbert%} proposed new recipe for training BERT by (a) training for longer with bigger batch size, (b) removing NSP task, (c) use longer sequences in training data format, i.e., use multiple sentences sampled contiguously to form longer segments, (d) change the masking pattern dynamically. Used BPE as in GPT-2. 

### T5: Text-to-Text Transfer Transformer

{%cite raffel2023exploringlimitstransferlearning%} adopts the natural language decathlon framework (in {%cite mccann2018naturallanguagedecathlonmultitask%} ) where many NLP tasks are translated into QA over a context.  Instead of an explicit QA format, T5 uses short task prefixes to distinguish task intentions and separately fine-tunes the model on every individual task.
The model is fine-tuned for each downstream task separately via “adapter layers” (add an extra layer for training) or “gradual unfreezing” (like ULMFit)


### GPT-3
{%cite brown2020languagemodelsfewshotlearners%} has the same architecture as GPT-2 but with 175B parameters (10x larger than GPT-2), 96 transformer layers.  In addition, GPT-3 uses alternating dense and locally banded sparse attention patterns, same as in sparse transformers. In order to fit such a huge model across multiple GPUs, GPT-3 is trained with partitions along both width and depth dimension.

### XLNet
{%cite yang2020xlnetgeneralizedautoregressivepretraining%}

Motivation: The Autoregressive (AR) model such as GPT and autoencoder (AE) model such as BERT are two most common ways for language modeling. However, each has their own disadvantages: AR does not learn the bidirectional context, which is needed by downstream tasks like reading comprehension and AE assumes masked positions are independent given all other unmasked tokens which oversimplifies the long context dependency.

Details are presented in {% cite weng2019LM%} but skipped since they seem wrong. 

### BART:  Bidirectional and AutoRegressive Transformer
{%cite lewis2019bartdenoisingsequencetosequencepretraining%} is a  denoising autoencoder to recover the original text from a randomly corrupted version jointly training BERT-like bidirectional encoder and GPT-like autoregressive decoder together. The best noising approach they discovered is text infilling and sentence shuffling (after experimenting with a variety of noising transformations).

{: .text-center}
![BART Comparison]({{ site.url }}{{ site.baseurl }}/docs/images/llm-bart.png)
Fig. A schematic comparison of BART with BERT {%cite devlin2019bertpretrainingdeepbidirectional%} and GPT {%cite radford2018improving%}. Credits: From {%cite lewis2019bartdenoisingsequencetosequencepretraining%} 

### ELECTRA: Efficiently Learning an Encoder that Classifies Token Replacements Accurately
{%cite clark2020electrapretrainingtextencoders%} aims to improve the pre-training efficiency, which frames the language modeling as a discrimination task instead of generation task. Proposes Replaced Token Detection (RTD).  Let’s randomly sample k positions to be masked. Each selected token in the original text is replaced by a plausible alternative predicted by a small language model, known as the generator G. The discriminator D predicts whether each token is original or replaced. The loss for the generator is the negative log-likelihood just as in other language models. The loss for the discriminator is the cross-entropy. Note that the generator is not adversarially trained to fool the discriminator but simply to optimize the NLL, since their experiments show negative results. After pretraining the generator is discarded and only the ELECTRA discriminator is fine-tuned further for downstream tasks.

* more beneficial to only share the embeddings between generator & discriminator while using a small generator (1/4 to 1/2 the discriminator size), rather than sharing all the weights (i.e. two models have to be the same size then)
* joint training of the generator and discriminator works better than two-stage training of each alternatively

## Recent Models

### Phi-Models

phi-1 {%cite gunasekar2023textbooksneed%} (1.3B parameters), phi-1.5 {%cite li2023textbooksneediiphi15%} (1.3B parameters), and phi-2 (2.7B parameters)

### Bloomberg-GPT
{%cite wu2023bloomberggptlargelanguagemodel%} 50-billion parameter language model for finance, trained on 363 billion tokens from finance data and 345 billion tokens from a general, publicly available dataset. For comparison, GPT-3 is 3.5x larger (175 billion parameters) but was trained on 1.4x fewer tokens (499 billion).
Why did the authors use an architecture with "only" 50 billion parameters since GPT-3 is 3.5x larger? That's easier to answer. They adopted the Chinchilla scaling laws and found this to be a good size given the available size of the finance data.
AdaptLLM-7B (Small) model outperforms BloombergGPT on one dataset and nearly matches its performance on three other finance datasets. Although BloombergGPT appears to be slightly better overall, it's worth noting that training AdaptLLM-7B cost about $100, in contrast to BloombergGPT's multi-million dollar investment.


### Llama Series

#### Llama 2
* 7B to 70B parameters
* Chat model versions are finetuned with RLHF using two separate reward models, rejection sampling and PPO
* Architecture: Group-query attention 

#### Llama 3
* 405B model, and updated the previous 8B and 70B models
* Uses group-query attention
* Didn’t use sliding attention window, and MoE approaches
* Supports 8 languages
* Employs heuristic-based filtering alongside model-based quality filtering, utilizing fast classifiers like Meta AI's fastText and RoBERTa-based classifiers. These classifiers also help in determining the context categories for the data mix used during training.
* Pre-training is performed in 3 stages
  * Standard Initial Pre-training
    * (4M tokens, 1024 batch size, and 4096 sequence length) => double the sequence length to 8192 after 252M tokens => double the batch size after 2.87T tokens. 
    * Adjusted the mix of data being used during the training process to optimize model learning and performance
  *  Continued Pre-training: Extended the context length from 8k to 128k in six distinct stages. Involved 8B tokens.
    * Annealing on High-Quality Data
* Post-Training: RLHF/DPO were iteratively repeated
  *  RM was trained using a checkpoint from the pre-training phase, utilizing human-annotated data. RM was further used for rejection sampling, helping to select appropriate prompts for further training. 
  * In each training round, model averaging was performed for RM, DPO and SFT


### Mistral 7B

{%cite jiang2023mistral7b%}

* Led to the development of Zephyr-7B and Mistral MoE suite of models  
* Outperforms 13B Llama-2 in various benchmarks.  
* Why it’s so good is unclear, but most likely due to its training data  
* Uses sliding window attention for computational efficiency  
  * Attention size block is 4096 tokens, trained with 100k token context sizes.

### Qwen 2

{%cite yang2024qwen2technicalreport%} from Alibaba Research

* 4 regular (dense) LLMs w/ 0.5B, 1.5B, 7B and 72B  
* MoE with 57B where 14B are activated at the same time  
* Multilingual capability in 30 languages  
* Focus on improving the data filtering pipeline to remove low-quality data and enhancing data mixing to increase data diversity   
* Performed training in two stages  
  * regular pre-training followed by long-context training  
  * long-context training increases the context length from 4096 to 32768 using high quality lengthy data  
* RLHF was done in 2 stages, first using DPO on an existing dataset (offline stage). Second, using a reward model to form the preference pair (online). Here, the model generates multiple responses during training, and a reward model selects the preferred response for the optimization step in "real-time" (that is, during training). This is also often referred to as "rejection sampling."

### Apple Intelligence Language Foundational Models (AFM)

{%cite gunter2024appleintelligencefoundationlanguage%}

* 3-billion-parameter on-device model intended for deployment on phones, tablets, or laptops, and a more capable server model of unspecified size.   
* Both models are dense, no MoE  
* Pre-training was performed in 3 stages  
  * Core Pre-Training: AFM-server model was trained on 6.3 trillion tokens, a batch size of 4096 batch size and a 4096-token sequence length. On-device model, which is distilled and pruned from a larger 6.4-billion-parameter model (trained from scratch like the AFM-server model).  
    * A distillation loss is used by replacing the target labels with a convex combination of the true labels and the teacher model's top-1 predictions (with 0.9 weight assigned to the teacher labels).  
  * Continued Pre-training: where web-crawl (lower-quality) data was down-weighted; math and code was up-weighted. Includes a small context lengthening step from 4,096 to 8,192 tokens on a dataset consisting of 1 trillion tokens.  
  * Context-lengthening: 100 billion tokens (10% of the tokens used in the second stage) but represents a more significant context lengthening to 32,768 tokens. To achieve this, the researchers augmented the dataset with synthetic long-context Q\&A data.  
* Post-Training  
  * leveraged both human-annotated and synthetic data, they fine-tuned the data mixture through multiple experiments to achieve the optimal balance. Introduced two new algorithms for RLHF: (a) Rejection Sampling Fine-tuning with Teacher Committee (iTeC), (b) RLHF with Mirror Descent Policy Optimization

{: .text-center}
![knowledge distillation]({{ site.url }}{{ site.baseurl }}/docs/images/llm-knowledge-distillation.jpg)
Fig. An overview of knowledge distillation, where a small model (here, the AFM-device 3B model) is trained on the original training tokens plus the outputs from a larger teacher model (here, a 6.4B model). Note that the cross entropy loss in a) is the regular training loss used for pre-training LLMs. Credits: From {%cite seb-new-llm-pretraining-posttraining%} 

**Gemma 2**

{%cite gemmateam2024gemma2improvingopen%}

* 2B, 9B and 27B, employs sliding window attention.  
* argue that even small models are often undertrained. However instead of increasing training dataset size, they maintain high quality, and use knowledge distillation methods.   
* 27B was trained from scratch, but 2B/9B were trained from knowledge distillation  
* Alternated between regular attention and sliding window attention layers. The sliding attention block size was 4096 tokens, spanning a total block size of 8192 tokens.  
* Performed logit capping, essentially a form of min-max normalizing and clipping of the logit values to keep them within a certain range (to improve stability and gradient flow during training). logits ← soft\_cap ∗ tanh(logits/soft\_cap)  
*   
* Post Training  
  * Used a mix of human-generated and synthetic-generated content. responses were primarily generated by teacher models, and knowledge distillation was also applied during the SFT phase.  
  * The reward model used for RLHF is ten times larger than the policy (target) model.  
  * Policy Models were averaged using WARP


## Comparing Open-Source Models

|  | Token Vocab size | Tokens for Training |
| :---- | :---- | :---- |
| Qwen 2 | 151642 | 7T (for 1.5B, 7B, 72B) 12T(for 0.5B) |
| Llama 2 | 32k | 2T |
| Llama 3.1 | 128k | 15T |
| AFM | 49k (device) 100k (server) | 6.3T (server) |
| Gemma 2 | 256k | 13T (27B) 8T (9B) 2T (2B) |
| Phi-3 | 32k |  |

Gemma 2 is likely the most capable model for single-GPU use cases today. For larger models, Llama 3 70B and Qwen 2 72B remain strong contenders {%cite seb-instruction-pretraining%}
{: .notice--info}

<!-- ## Reading List


| Title                                          |  Topic       |   Comments                                                   |
| --------------------------------------------   | ------------ | ------------------------------------------------------------ |
| [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) {% cite alammar-illustratedtbert%}| LMs | Good Short Overview
| [Generalized Language Models by Lilian Weng](https://lilianweng.github.io/posts/2019-01-31-lm/) {% cite weng2019LM%}| LMs | Great overview of BERT and its successors
| [Ten Noteworthy AI Research Papers of 2023 by Sebastian Raschka](https://magazine.sebastianraschka.com/p/10-ai-research-papers-2023) {% cite seb-10aipapers2023%}| LMs/Research | Decent samplers of 2023 10 papers
| [AI and Open Source in 2023](https://magazine.sebastianraschka.com/p/ai-and-open-source-in-2023) {% cite seb-ai-opensource%}| LMs/Research | Decent samplers of 2023 10 papers
| [New LLM Pre-training and Post-training Paradigms](https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training) {% cite seb-new-llm-pretraining-posttraining%}| LMs/Training/Research | detailed overview of pre-training pipelines -->


## References


{% bibliography --cited %}