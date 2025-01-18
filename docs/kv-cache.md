---
layout: single
title: KV-Cache
permalink: /docs/kv-cache/
last_modified_at: 2025-01-17T17:00:00-08:00s
toc: true
---

Since the inference with transformer based models is an autoregressive process, and has causal self-attention (only self and past tokens are considered to predict the next token), we only need to compute the representation for each new token, but other tokens remain fixed (i.e., because they don’t depend on tokens that follow them).

The KV cache stores the linear projection of Key and Value matrix, i.e., $$K$$ and $$V$$ (resulted after multiplication from $$W^k$$ and $$W^v$$ respectively). At every time step, when we have the new input token (based on what token was sampled in the previous timestep), we only need to compute the query projection of the new token (by multiplying with $$W^q$$) and won’t need the query projection of all previous tokens. Now, we use the query projection for the new token and update and add the new row in KV cache.

KV-caching decreases the latency to the next token in an autoregressive setting starting from the second token. Since the prompt tokens are not cached at the beginning of the generation, time to the first token is high, but as KV-caching kicks in for further generation, latency reduces.
