---
layout: single
title: Large Multimodal Models (LLMs)
permalink: /docs/large-multimodal-models/
toc: true
---

## Taxonomy of MLMs

Training/Finetuning multi-modality LMs is similar like VLMs, and hence the major focus in terms of understanding MLMs is VLMs.
{: .notice--info}

Typically, training VLMs is roughly categorized into four buckets

1. Translating images into embedding features that can be jointly trained with token embeddings,   
   * Treat images as normal text tokens and train the model on a sequence of joint representations of both text and images. Precisely, images are divided into multiple smaller patches and each patch is treated as one *token* in the input sequence.   
   * Examples: VisualBERT {%cite li2019visualbertsimpleperformantbaseline%}, SimVLM {%cite wang2022simvlmsimplevisuallanguage%}, CM3 {%cite aghajanyan2022cm3causalmaskedmultimodal%} 
2. Learning good image embeddings that can work as a prefix for a frozen, pre-trained language model.  
   * Use vision encoders (like Resnets) and update only the Vision Encoder parameters while keeping the LM frozen. Trained on ALIGN {%cite jia2021scalingvisualvisionlanguagerepresentation%} datasets.  
   2. Examples: Frozen {%cite tsimpoukelli2021multimodalfewshotlearningfrozen%}, ClipCap {%cite mokady2021clipcapclipprefiximage%}
3. Text-Image Cross Attention and Fuse Mechanisms  
   1. Examples: ViLBERT {%cite lu2019vilbertpretrainingtaskagnosticvisiolinguistic%}, VisualGPT {%cite chen2022visualgptdataefficientadaptationpretrained%}, VC-GPT {%cite luo2022frustratinglysimpleapproachendtoend%}, MERLOT (on videos) {%cite zellers2021merlotmultimodalneuralscript%}, FLAMINGO {%cite alayrac2022flamingovisuallanguagemodel%}, CoCa {%cite yu2022cocacontrastivecaptionersimagetext%}
4. Combine vision and language models without any training.   
   1. MAGiC {%cite su2022languagemodelsseeplugging%} performs guided decoding using CLIP-based score without finetuning, has decent performance compared to other unsupervised approaches, but still has big gaps with supervised methods.  
   2. PiCA {%cite yang2022empiricalstudygpt3fewshot%} first converts the images into captions or tags and then uses few-shot examples to prompt GPT3 to provide answers.  
   3. Socratic Models {%cite zeng2022socraticmodelscomposingzeroshot%}  is a framework to compose multiple pretrained models for different modality via language (prompting) into one model without further training.  
5. Others (not sure if they are part of above?)  
   1. Contrastive Learning: CLIP {%cite radford2021learningtransferablevisualmodels%}, ALIGN  {%cite jia2021scalingvisualvisionlanguagerepresentation%}
   2. FLORENCE? (may be still contrastive learning) {%cite yuan2021florencenewfoundationmodel%}

## Models

### VisualBERT
{%cite li2019visualbertsimpleperformantbaseline%} feeds both text inputs and image regions into BERT such that it is able to discover the internal alignment between images and text with self-attention mechanisms. Each visual embedding in VisualBERT also sums up three types of embeddings:
* tokenized features ($$f_o$$) is a visual feature vector of the bounding region produced by a CNN
* segmentation embedding ($$f_s$$) to indicate whether the embedding is for vision not for text;
* position embedding ($$f_p$$) used for aligning the order of bounding regions.

The model is trained on MS COCO image caption dataset with both text and image as inputs to predict text captions, using two visually-grounded language model objectives:
* **MLM with the image** The model needs to predict masked text tokens, while image embeddings always stay not masked.
* **Sentence-image prediction** When provided with an image and two associated captions, one of two captions might be a random unrelated caption with 50% probability. The model is asked to distinguish these two situations.

According to ablation experiments, the most important configuration is to fuse visual information early on into the transformer layers and to pretrain the model on the COCO caption dataset.

{: .text-center}
![VisualBERT Architecture]({{ site.url }}{{ site.baseurl }}/docs/images/lmm-visualbert.png)
Fig. The architecture of VisualBERT {%cite li2019visualbertsimpleperformantbaseline%} 

### SimVLM: Simple Visual Language Model
{%cite wang2022simvlmsimplevisuallanguage%} is a simple prefix language model, where the prefix sequence is processed with bi-directional attention like BERT, but the main input sequence only has causal attention like GPT. Images are encoded as prefix tokens such that the model can fully consume the visual information and then generates associated text in an autoregressive manner. Inspired by ViT and CoAtNet, SimVLM splits the image into smaller patches in a flatten 1D sequence of patches. They use the convolutional stage consisting of the first 3 blocks of ResNet to extract contextualized patches and this setup is found to work better than a naive linear projection. Training data contains ALIGN and text-only data from C4. According to ablation studies, it is important to have both image-text and text-only data for training.

Utilizes a version of the vision transformer (ViT). In fact, they replaced the well-known patch projection with three ResNet blocks to extract image patch vectors (Conv stage in the image below). The ResNet blocks are trained together with the entire model, contrary to other methods where a fully-pretrained image module is used. 

{: .text-center}
![SimVLM Model]({{ site.url }}{{ site.baseurl }}/docs/images/lmm-simvlm.png)
Fig. Illustration of the SimVLM model. This shows an example of training with PrefixLM
of an image-text pair. For text-only corpora, it is straightforward to remove the image patches and
utilize textual tokens only. {%cite wang2022simvlmsimplevisuallanguage%}

### CM3: Causally-Masked Multimodal Modeling
{%cite aghajanyan2022cm3causalmaskedmultimodal%} is a hyper-text language model, learning to generate the content (hypertext markup, hyperlinks and images) of large scale HTML web pages of CC-NEWS and Wikipedia articles. The resulting CM3 models can generate rich structured, multi-modal outputs while conditioning on arbitrary masked document contexts. Architecture-wise, CM3 is an autoregressive model. However, in order to combine causal and masked language modeling, CM3 also masks out a small number of long token spans and tries to generate them at the end of the sequences.

{: .text-center}
![CM3 Modeling Objective]({{ site.url }}{{ site.baseurl }}/docs/images/lmm-cm3.png)
Fig. A visual representation of various language modeling objectives as well as our proposed
causal language modeling objective with a single mask (n = 1). Given the left-to-right nature of
causal language models (bottom row) we would not be able to generate the Wikipedia entity link
highlighted in orange. Credits: {%cite aghajanyan2022cm3causalmaskedmultimodal%}

### ClipCap
{%cite mokady2021clipcapclipprefiximage%} relies on CLIP for vision encoding, and learns a network F mapping CLIP embeddings into a sequence of k embedding vectors (with similar dimension as word token dimension in GPT-2). Both CLIP and LM is frozen during training, and only the mapping network $$F$$ is learned. 
* Increasing the prefix size $$k$$, improves the performance
* They found that when LM is frozen, $$F$$ should be a transformer, with 8 multi-head self-attention layers with 8 heads each, but when LM can be fine-tuned, a MLP is enough
* The fun fact is - because ClipCap translates CLIP image embeddings into LM space, the processed prefixes can be even interpreted as words.


### ViLBERT
{%cite lu2019vilbertpretrainingtaskagnosticvisiolinguistic%} used co-attention module to calculates importance scores based on both images and text embeddings.

{: .text-center}
![ViLBERT]({{ site.url }}{{ site.baseurl }}/docs/images/lmm-vilbert.png)
Fig. ViLBERT model consists of two parallel streams for visual (green) and linguistic
(purple) processing that interact through novel co-attentional transformer layers. This structure allows
for variable depths for each modality and enables sparse interaction through co-attention. Dashed
boxes with multiplier subscripts denote repeated blocks of layers. Credits: {%cite aghajanyan2022cm3causalmaskedmultimodal%}

### VisualGPT

{%cite chen2022visualgptdataefficientadaptationpretrained%} introduced a self-resurrecting activation unit (SRAU) to control the tradeoff between a mixture of pre-trained linguistic information and visual components. 

### VC-GPT: Visual Conditioned GPT 
{%cite luo2022frustratinglysimpleapproachendtoend%} combines a pretrained visual transformer (CLIP-ViT) as visual encoder and a pretrained LM as language decoder. The CLIP-ViT takes a sequence of image patches as inputs and outputs representation for each patch. To avoid catastrophic forgetting, instead of injecting the visual information directly into GPT2, VC-GPT introduces extra cross-attention layers on top of the output of visual encoder and language decoder. Then a self-ensemble module linearly combines the single model language decoder logits  and cross-model vision-language fused module logits. The self-ensemble module is important for the performance.

{: .text-center}
![VC-GPT]({{ site.url }}{{ site.baseurl }}/docs/images/lmm-vcgpt.png)
Fig. : Comparing the differences between the Vanilla Framework and our VC-GPT Framework. Credits: {%cite luo2022frustratinglysimpleapproachendtoend%}

### CLIP
{%cite radford2021learningtransferablevisualmodels%} Key Highlights

* First model that could generalize to multiple image classification tasks with zero- and few-shot learning.  
* CLIP’s key contribution is its ability to map data of different modalities, text and images, into a shared embedding space.  
* Flamingo and LLaVa use CLIP as their image encoder. DALL-E uses CLIP to rerank generated images.  
* CLIP leveraged natural language supervision and contrastive learning, which allowed CLIP to both scale up their data and make training more efficient.

**Architecture**


{: .text-center}
![CLIP]({{ site.url }}{{ site.baseurl }}/docs/images/lmm-clip-architecture.png)
Fig. : CLIP Summary. While standard image models jointly train an image feature extractor and a linear classifier to predict
some label, CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of (image, text) training
examples. At test time the learned text encoder synthesizes a zero-shot linear classifier by embedding the names or descriptions of the
target dataset’s classes Credits: {%cite radford2021learningtransferablevisualmodels%}

**Image Encoder**

Experimented with both ResNet and ViT. Their best-performing model is ViT-L/14@336px

* Large vision transformer (ViT-L)  
* 14 patches (each image is divided into 14x14 pixel patches/sub-images)  
* on 336 x 336 pixel input

**Text Encoder**

Uses a transformer model similar to GPT-2, but smaller. Their base model has only 63M parameters with 8 attention heads. The authors found CLIP’s performance to be less sensitive to the capacity of the text encoder.

**Alignment**

Embeddings generated by the image encoder and text encoder are projected into the same embedding space using two projection matrices Wv and Wl.

* Given an image embedding $$V_i$$, the corresponding multimodal embedding is computed as $$W_vV_i$$.  
* Given a text embedding $$L_i$$, the corresponding multimodal embedding is computed as: $$W_lL_i$$.

**Natural Language Supervision**

Previously, models (e.g., ImageNet, MS COCO) were trained with manually annotated datasets, i.e., (image, text) pairs, which isn’t scalable since it’s time consuming and expensive. They created their own dataset – 400M (image, text) pairs – as follows.

* Construct a list of 500k queries. Queries are common words, bigrams, and titles of popular Wikipedia articles.  
* Find images matching these queries (string and substring match). The paper mentioned this search did NOT happen on search engines but didn’t specify where. My theory is that since OpenAI already scraped the entire Internet for their GPT models, they probably just queried their internal database.  
* Each image is paired with a text that co-occurs with it (e.g. captions, comments) instead of the query since queries are too short to be descriptive.

Because some queries are more popular than others, to avoid data imbalance, they used at most 20K images for a query.

**Language Model Objective**

If a classifier outputs only one class for each input, a language model outputs a sequence of classes. Each generated class is called a token. Each token is from a predetermined list, the vocabulary, of the language model. While the language model objective allows for vastly more flexible outputs, CLIP authors noted this objective made the training difficult. They hypothesized that this is because the model tries to generate exactly the text accompanying each image, while many possible texts can accompany an image: alt-text, caption, comments, etc. For example, in the Flickr30K dataset, each image has 5 captions provided by human annotators, and the captions for the same image can be very different.

**Contrastive Learning**

Instead of predicting the exact text of each image, CLIP was trained to predict whether a text is more likely to accompany an image than other texts. For each batch of $$N$$ (image, text) pairs, the model generates $$N$$ text embeddings and $$N$$ image embeddings.
* Let $$V_1, V_2, \cdots, V_n$$ be the embeddings for the $$N$$ images.  
* Let $$L_1, L_2, \cdots, L_n$$ be the embeddings for the $$N$$ texts.

CLIP computes the cosine similarity scores of the $$N^2$$ possible $$(V_i, L_j)$$ pairings. The model is trained to maximize the similarity scores of the $$N$$ correct pairings while minimizing the scores of the $$N^2 - N$$ incorrect pairings. For CLIP, $$N = 32,768$$.

Another way to look at this is that each training batch of CLIP is two classification tasks.

1. Each image can be paired with $$N$$ possible texts, and the model tries to predict the correct one. This is the same setup as image-to-text retrieval.   
   $$L_{contrastive:txt2im} = -\frac{1}{N}\sum_i^Nlog \frac{exp(L_i^TV_i\beta)}{\sum_j^Nexp(L_i^TV_j\beta)}$$  
2. Each text can be paired with $$N$$ possible images, and the model tries to predict the correct image. This is the same setup as text-to-image retrieval.  
   $$L_{contrastive:im2txt} = -\frac{1}{N}\sum_i^Nlog \frac{exp(V_i^TL_i\beta)}{\sum_j^Nexp(V_j^TL_i\beta)}$$  

The sum of these two losses is minimized. 𝛽 is a trainable inverse temperature parameter.

CLIP authors found that the contrastive objective provided a 12x improvement in efficiency compared to the language model objective baseline while producing higher-quality image embeddings.

**Applications (Beyond Classification)**

* Text-based Image Retrieval  
  * Generate CLIP embeddings for all your images and store them in a vector database.  
  * For each text query, generate a CLIP embedding for this text.  
  * Query in the vector database for all images whose embeddings are close to this text query embedding.  
* Image Generation: unCLIP  
  * Given a text prompt, DALL-E (2021) generates many different visuals and uses CLIP to rerank these visuals before showing the top visuals to users.  
  * In 2022, OpenAI introduced unCLIP, a text-to-image synthesis model conditioned on CLIP latents. It consists of two main components:  
    * CLIP is trained and frozen. The pretrained CLIP model can generate embeddings for both text and images in the same embedding space. Two things happen at image generation (a) Use CLIP to generate embedding for this text. (b) Use a diffusion decoder to generate images conditioned on this embedding.  
* Text Generation  
  * CLIP's text generation experiments, including the LM RN50 model, performed about 10% worse than their best model on vision-language tasks. Today, while CLIP itself isn't used for text generation, its image encoder serves as a foundation for many language-vision models.


### FLAMINGO

Flamingo {%cite alayrac2022flamingovisuallanguagemodel%} can generate text responses conditioned on both text and images. In a reductive view, Flamingo is CLIP + a language model, with added techniques to make it possible for the language model to generate text tokens conditioned on both visual and text inputs.

**Architecture**

{: .text-center}
![FLAMINGO]({{ site.url }}{{ site.baseurl }}/docs/images/lmm-flamingo-architecture.png)
Fig. : Flamingo architecture overview. Credits: {%cite alayrac2022flamingovisuallanguagemodel%}

**Vision encoder**

* First, a CLIP-like model is trained using contrastive learning from scratch. The text encoder of this model is then discarded. The vision encoder is frozen to be used in the main model.  
* Uses the 2 (image, text) pair datasets, ALIGN and LTIP, totaling 2.1B (image, text) pairs. 5x larger than the CLIP dataset.  
* For the text encoder (of the CLIP part), Flamingo uses BERT instead of GPT-2. For the vision encoder (of the CLIP part), Flamingo uses a NormalizerFree ResNet (NFNet) F6 model. Text and vision embeddings are meanpooled before being projected to the joint embedding space.

**Language model**

* Flamingo finetunes chinchilla to generate text tokens (specifically, they freeze 9 pre-trained Chinchilla LM layers), conditioned on visuals and text, using language model loss, with two additional components Perceived Resampler and GATED XATTN-DENSE layers. More details about these layers are in {% cite chip-multimodal%}

Perceiver-based architecture produces a few hundreds of tokens out of a large number of visual input features and then uses cross-attention layers interleaved with the LM layers to fuse visual information into the language decoding process.

Similar to ClipCap, both pretrained models are frozen during training and thus Flamingo is only trained to harmoniously connect existing, powerful language and vision models together. The main difference between ClipCap and Flamingo is that the former treats the image embedding as a simple prefix for LM, while the latter uses the gated cross-attention-dense layer to fuse image information. In addition, Flamingo incorporates a lot more training data than ClipCap.

Dataset: Flamingo used 4 datasets,

* 2 (image, text) pair datasets: ALIGN and LTIP  
* 1 (video, text) pair dataset: VTP  
* 1 interleaved image and text dataset: M3W (built themselves)  
  * The input Web page text is processed by inserting \<image\> tags at the location of visual inputs, as well as special tokens, \<BOS\> (beginning of sentence) and \<EOC\> (end of chunks; always at the end of the document, before any image tag).

Masking in Flamingo is designed such that text tokens only cross-attends to visual tokens corresponding to the last preceding image, largely reducing the number of visual tokens that a certain text token can see. They found this works better than allowing text tokens to attend to all preceding images directly. Text still can attend to all previous images because there is a causal self-attention dependency in the text encoder. This design can deal with an arbitrary number of images in the context

Since FLAMINGO is trained on a mixture of datasets, it optimizes for a weighted sum of dataset specific NLL losses. In practice, instead of round-robin between datasets, they actually sample one batch from each dataset and apply a weighted sum of these gradients in each update. 

### CoCa: Contrastive Captioning

{%cite yu2022cocacontrastivecaptionersimagetext%} captures both the merits of contrastive learning and image-to-caption generation, jointly trained from scratch (using ALIGN and JTB-3B) with contrastive loss on CLIP-style representation and generative loss on image captioning. It combines contrastive learning, and encoder-decoder captioning loss. 

The bottom unimodal component encodes the input text with causally-masked self-attention. The top multimodal component applies both causally-masked self-attention and cross-attention to the output of the vision encoder. They use task-specific attention pooling, or attention pooler, as a natural task adapter, as they found that a single pooled image embedding helps visual recognition tasks (e.g. ImageNet classification), while a more fine-grained embedding helps multimodal understanding tasks (e.g. VQA).  A pooler is a single multi-head attention layer with n learnable queries with the encoder output as both keys and values. CoCa uses attentional poolers in pretraining for generative loss $$n=256$$ and contrastive loss $$n=1$$.

{: .text-center}
![CoCa pretraining]({{ site.url }}{{ site.baseurl }}/docs/images/lmm-coca-pretraining.png)
Fig. : Overview of Contrastive Captioners (CoCa) pretraining as image-text foundation models.
The pretrained CoCa can be used for downstream tasks including visual recognition, vision-language
alignment, image captioning and multimodal understanding with zero-shot transfer, frozen-feature
evaluation or end-to-end finetuning. Credits: {%cite yu2022cocacontrastivecaptionersimagetext%}

### ALIGN
{%cite jia2021scalingvisualvisionlanguagerepresentation%} uses a dual-encoder that learns to align visual and language representations of image-text pairs using contrastive loss similar to CLIP (im2txt and txt2im). Training is performed with a noisy dataset of one billion image-text pairs. So instead of doing expensive preprocessing on the data as similar methods do, they show that the scale of the dataset can compensate for the extra noise

### FLORENCE
{%cite yuan2021florencenewfoundationmodel%} proposes end-to-end learning for VL tasks (as a foundational model).

* Hierarchical vision transformer (Swin) as the image encoder and a modified CLIP as the language decoder  
* The training is performed on “image-label-description” triplets.  
* bidirectional contrastive learning: the loss contains two contrastive terms; an image-to-language contrastive loss and a language-to-image contrastive loss. In a way, they try to combine two common learning tasks: the mapping of images to the labels and the assignment of a description to a unique label.  
* They enhance the pretrained representations into more fine-grained representations with the use of “adapter” models.

## VL Generative Models

### Dall-E

{%cite ramesh2021zeroshottexttoimagegeneration%} uses a discrete variational autoencoder to map the images to image tokens. dVAE essentially uses a discrete latent space compared to a typical VAE. The text is tokenized withBPE. The image and text tokens are concatenated and processed as a single data stream. DALL-E uses an autoregressive transformer to process the stream in order to model the joint distribution of text and images. In the transformer’s decoder, each image can attend to all text tokens. At inference time, we concatenate the tokenized target caption with a sample from the dVAE, and pass the data stream to the autoregressive decoder, which will output a novel token image.


### GLIDE
{%cite nichol2022glidephotorealisticimagegeneration%} Diffusion Based

## Reading List

| Title                                          |  Topic       |   Comments                                                   |
| --------------------------------------------   | ------------ | ------------------------------------------------------------ |
| [Multimodality and Large Multimodal Models (LMMs) by Chip Huyen](https://huyenchip.com/2023/10/10/multimodal.html) {% cite chip-multimodal%}| MMs | Great review of MMs, with CLIP , FLAMINGO and insights
| [Generalized Visual Language Models by Lilian Weng](https://lilianweng.github.io/posts/2022-06-09-vlm/) {% cite weng2022vlm%}| MMs | Great overview of VLM techniques
| [Primers - Vision Language Models](https://aman.ai/primers/ai/vision-language-models/) {% cite Chadha2020DistilledVisionLanguageModels%}| MMs | Average Read

## References


{% bibliography --cited %}