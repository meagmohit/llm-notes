---
layout: single
title: Vision Transformers
permalink: /docs/vision-transformers/
last_modified_at: 2025-01-17T17:00:00-08:00s
toc: true
---

* Motivation: In CNNs, you start off being very local and slowly get a global perspective. A CNN recognizes an image pixel by pixel, identifying features like edges, corners, or lines by building its way up from the local to the global. But in transformers, owing to self-attention, even the very first attention layer models global contextual information, making connections between distant image locations (just as with language). If we model a CNN’s approach as starting at a single pixel and zooming out, a transformer slowly brings the whole fuzzy image into focus.
* ViT model is very identical to the original transformers {%cite vaswani2017attention%}, with a few changes to adapt it to the vision domain without excessively increasing the computing times - divide the large images into square units (or patches, similar to tokens).
* Pretrained ViTs are even easier to finetune than convolutional neural networks.
* {%cite smith2023convnetsmatchvisiontransformers%} CNNs are competitive with ViTs given access to the large enough datasets.

## References


{% bibliography --cited %}