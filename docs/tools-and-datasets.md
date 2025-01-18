---
layout: single
title: Tools & Datasets
permalink: /docs/tools-and-datasets/
last_modified_at: 2025-01-17T17:00:00-08:00s
toc: true
---

## LLM (Language) Tasks and Datasets
**QA**
* SQuAD: Stanford QA Dataset
* RACE: Reading Comprehension from Examinations

**Commonsense Reasoning**
* Story Cloze Test: 
* SWAG: Situations With Adversarial Generations

**Natural Language Inference (NLI) / Text Entailment**
Goal: Whether one sentence can be deferred from another
* RTE (Recognizing Textual Entailment)
* SNLI (Stanford Natural Language Inference)
* MNLI (Multi-Genre NLI)
* QNLI (Question NLI)
* SciTail

**Named Entity Recognition (NER)**
* CoNLL 2003 NER task
* OntoNotes 5.0
* Reuter Corpus
* Fine-Grained NER (FGN)

**Sentiment Analysis**
* SST (Stanford Sentiment Treebank)
* IMDb

**Semantic Role Labeling (SRL)**
Goal: models the predicate-argument structure of a sentence, and is often described as answering “Who did what to whom”.
* CoNLL-2004 & CoNLL-2005
**Sentence similarity (or paraphrase detection)**
* MRPC (Microsoft Paraphrase Corpus)
* QQP (Quora Question Pairs) STS Benchmark

**Sentence Acceptability**
A task to annotate sentences for grammatical acceptability
* CoLA (Corpus of Linguistic Acceptability)

**Text Chunking**
To divide a text in syntactically correlated parts of words.
* CoNLL 2000

**Part-of-Speech (POS) Tagging**
* Wall Street Journal portion of the Penn Treebank

**Machine Translation**
* WMT 2015 English-Czech data (Large)
* WMT 2014 English-German data (Medium)
* IWSLT 2015 English-Vietnamese data (Small)

**Coreference Resolution**
Cluster mentions in text that refer to the same underlying real world entities.
* CoNLL 2012

**Long-range Dependency**
* LAMBADA (LAnguage Modeling Broadened to Account for Discourse Aspects)
* Children’s Book Test

**Multi-task benchmark**
* GLUE multi-task benchmark
* decaNLP benchmark

**Unsupervised pretraining dataset**
* Books corpus
* 1B Word Language Model Benchmark
* English Wikipedia

## LMM (Multimodal) Tasks and Datasets

**Image Caption Datasets**
  * MS COCO  
  * NoCaps   
  * Conceptual Captions  
  * Crisscrossed Captions (CxC)  
  * Concadia  

**Pair Image Text Datasets**
  * ALIGN  
  * LTIP\*  
  * VTP\*  
  * JFT-300M/JFT-3B\*  

**VQA**
Refers to the process of providing an answer to a question given a visual input (image or video).  
  * VQAv2  
  * OkVQA  
  * TextVQA  
  * VizWiz  

**Visual Language Reasoning**
Infers common-sense information and cognitive understanding given a visual input.  
  * VCR (Visual Commonsense Reasoning)  
  * NLVR2 (Natural Language for Visual Reasoning)  
  * Flickr30K  
  * SNLI-VE (Visual Entailment)  

**Video QA and Understanding**
  * MSR-VTT (MSR Video to Text)  
  * ActivityNet-QA  
  * TGIF (Tumblr GIF)  
  * LSMDC (Large Scale Movie Description Challenge)  
  * TVQA/+  
  * DramaQA  
  * VLEP (Video-and-Language Event Prediction)

(\*) Internal, non-public datasets

### Common Pre-training Strategies

* **Masked Language Modeling** is often used when the transformer is trained only on text. Certain tokens of the input are being masked at random. The model is trained to simply predict the masked tokens (words)  
* **Next Sequence Prediction** works again only with text as input and evaluates if a sentence is an appropriate continuation of the input sentence. By using both false and correct sentences as training data, the model is able to capture long-term dependencies.  
* **Masked Region Modeling** masks image regions in a similar way to masked language modeling. The model is then trained to predict the features of the masked region.  
* **Image-Text Matching** forces the model to predict if a sentence is appropriate for a specific image.  
* **Word-Region Alignment** finds correlations between image region and words.  
* **Masked Region Classification** predicts the object class for each masked region.  
* **Masked Region Feature Regression** learns to regress the masked image region to its visual features.

## RLHF Datasets

* Anthropic RLHF Dataset on HuggingFace [[Dataset]](https://huggingface.co/datasets/Anthropic/hh-rlhf?row=98)
* No Robots of 10k instructions from HuggingFace, based on SFT dataset from InstructGPT [[Dataset]](https://huggingface.co/datasets/HuggingFaceH4/no_robots)

## 15K Token FineWeb Dataset

{%cite penedo2024finewebdatasetsdecantingweb%} describes, creates and makes a 15T token dataset publicly available. Based on Chinchilla Scaling Laws, the 15T dataset should be optimal for 500B parameters. Note that, RedPajama contains 20 trillion tokens, but the researchers found that models trained on RedPajama result in poorer quality than FineWeb due to the different filtering rules applied. The Llama 3 models with 8B, 70B, and 405B sizes were trained on 15 trillion tokens as well, but Meta AI's training dataset is not publicly available. 



## References

{% bibliography --cited %}