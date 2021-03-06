---
layout: post
title:  Decoding strategies in text generation
date:   2020-10-16 15:01:35 +0800
description: introduce decoding strategies for text generation task
img: 02.jpg
tags: [nlp, transformer, deep-learning] # add tag
---


## Before delving into the details...
Hi, this post is aimed at introducing common decoding strategies for text generation (ex: GPT-2). Decoding strategies play an important role when developing NLG (Natural Language Generation)models, and the choice of decoding algorithm could have huge impact on the quality of generated sequences!

In this post, I will explain what is auto-regressive language generation first, and then I will move on to the introduction of several decoding algorithms.

## Auto-regressive language generation
Auto-regressive language generation assumes that the element  of output sequence at timestep $$t$$ is determined by the input sequence and timesteps before $$t$$.

![](https://i.imgur.com/JKrMxhT.png)

where $$W_0$$ is the input sequence; $$W_t$$ is the word at timestep $$t$$; $$T$$ is determined by the position of <EOS> token.


Thanks to the come out of transformer, a well-knowned seq2seq model architecture, researchers now are able to well deal with lots of NLP problems, e.g. neural machine translation (NMT), text summerization, text generation. In terms of conducting text generation with transformer, decoding strategies play an important role and thus cannot be ignored. The easiest approach is generating tokens with the highest probabilities in each timestep.

## Decoding strategies in text generation
#### Beam search (BS)

Beam search (BS) is a common-used algorithm for decoding sequences. When decoding, BS consider the probabilty of whole sequence instead of just considering the word with highest probability at each timestep (greedy search). In other words, BS will maintein $$k$$ sequences (k is the beam width, you can define it on your own) at the same time, and when there is a new sequence coming, the sequence with lowest probability will be discarded.

But BS still has some drawbacks:
- It generates similar sequences and thus lost diversity.
- It is computationally wasteful since your inference time would be amplified by the beam width.
- With higher beam width, NMT tends to generate shorter sequences (since <EOS> token is more likely to be generated).

To avoid the repetition of generated sequences, a simple remedy called "n-gram penalty" has come to the stage. The n-gram penalty make sure that there is no n-gram showing up for more than once. That is to say, if there is a generated n-gram has been put into the output sequence before, then its probability would be set to zero.

#### Random sampling
In order to make the output sequences more surprising (which means more similar to humans' behavior), we need to add some randomness into the decoding process.

Instead of picking tokens with the highest probability, we sample from the distribution of tokens at each timestep.

Besides, we can add a parameter called "Temperature" (range from 0~1) to adjust the distribution of tokens.

{% highlight python %}

# Temperature
# make higher p much higher ; make lower p much lower
# _scores: distribution of tokens
# size of _scores: (bacth_size*num_beams, vocab_size)

if temperature != 1.0:
    _scores = _scores / temperature

{% endhighlight %}

#### Top-K/Top-P Sampling (Nucleus Sampling)


With sampling, we may face the chance of selecting "weird/improper words" (words with low probability) as our generating tokens. To solve this problem, top-K sampling is introduced to look at tokens with top-K probabilities. That is to say, only top $$K$$ tokens will be seen as the candidates of the generating sequence. After that, we can sample a token from the distribution formed by the $$K$$ tokens!

Nevertheless, we cannot dynamically change the value of $$K$$ case by case. Thus, top-P sampling come to solve this problem by selecting token candidates until their cumulatiove probability achieves a given $$P$$. Via top-P sampling, we now can select different amounts of token candidates based on different situations!



### Reference
- [how to generate from huggingface](https://huggingface.co/blog/how-to-generate)
- [github of Transformers](https://github.com/huggingface/transformers)

***
<small>image from [here](https://unsplash.com/photos/s9CC2SKySJM) </small>
