---
layout: single
title: Tokenization
permalink: /docs/tokenization/
toc: true
---

Per [OpenAI’s Tokenizer Platform](https://platform.openai.com/tokenizer) page, a helpful rule of thumb is that one token generally corresponds to ~4 characters of text for common English text. This translates to roughly 34 of a word (so 100 tokens ~= 75 words).

**Why not tokenize to simple words?**
For instance, English dictionaries have various forms, like plural (-s) or verbs (-ing), possessives and conjugations. Further, the data will have typographical errors (or else it needs to be cleaned thoroughly).  

## BPE (Byte Pair Encoding)
* **Encoding**: Replace the most frequent occurring byte pair with a byte that is not previously used in the data. Repeat until fix vocabulary size or when there are no new bytes added.
* **Decoding**: Perform the replacements in the reverse order. 

Since it retains all the byte codes for its single character building blocks, it can still represent weird misspellings, new words, and even foreign languages.

## References


{% bibliography --cited %}