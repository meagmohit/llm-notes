---
layout: single
title: Evaluation Overview
permalink: /docs/evaluation-overview/
last_modified_at: 2025-01-17T17:00:00-08:00s
toc: true
---


## Overview

### Human Preferences
Human preferences are used to rank the models (as used by LMSYS’s Chatbot Arena). The two most common ranking algorithms are Elo (from chess) and TrueSkill (from video games). While Chatbot Arena refers to their model scores “Elo scores”, they actually don’t use Elo. In December 2023, they switched to Bradley-Terry but scaled the resulting scores to make them look Elo-like. Given a history of match outcomes, the Bradley-Terry algorithm finds the model scores that maximize the likelihood of these match outcomes, turning model scoring into a maximum likelihood estimation problem. {%cite chip-rlhf%}


### Model Ranking
Given the same match outcomes, different ranking algorithms can produce different rankings. For example, the ranking computed by Elo might differ from the ranking computed by Bradley-Terry. How do we know that a ranking is correct?

At its core, model ranking is a predictive problem. We compute a ranking from historical match outcomes and use it to predict future match outcomes. The quality of a ranking is determined by how accurately it can predict future match outcomes.

## References

{% bibliography --cited %}