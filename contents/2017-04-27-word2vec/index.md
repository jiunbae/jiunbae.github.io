---
title: Word Embedding
description: word embedding for clustering
date: 2017-04-27
slug: /word2vec
tags: [nlp, tech]
heroImage:
heroImageAlt:
---
## NLP

NLP(Natural Language Processing)는 다음과 같은 질문에서 출발했습니다.

> "어떻게 하면 컴퓨터가 텍스트를 이해할 수 있을까?"

사실 아직까지도 컴퓨터는 우리의 머리 속에서 하는 것처럼 텍스트를 이해하지는 못합니다. 
대신 확률과 통계, 그리고 많은 수학을 토대로 비슷하게 작동하도록 만들 수는 있습니다.

그러면 우리는 어떻게 컴퓨터에게 텍스트의 의미를 알려줄 수 있을까요?

> “You shall know a word by the company it keeps”[^1]

물론 많은 방법이 있을 수 있겠지만, 오늘은 문맥을 통해서 텍스트를 이해시키는 방법에 대해 알아보고자 합니다.

## Word Embedding

컴퓨터가 텍스트를 이해할 수 있게 하기 위해 가장 먼저 해야할 일은 자연 언어를 수치적인 방식으로 표현하는 것입니다. 
하지만 컴퓨터에게 텍스트의 의미를 알려주는 것은 매우 어려운 일입니다. 
간단한 단어 두개의 개념적 차이와 각각의 의미를 어떻게 알려줄 수 있을까요?

이전에는 이를 해결하기 위해 one-hot encoding방식을 사용해 왔습니다. 
크기가 n인 단어 사전을 만들고 어떤 단어를 길이가 n인 단어 사전의 해당 단어가 아닌 단어들을 0으로 표시한 벡터로 만드는 방법입니다. 
이런 방식은 단어 간의 관계를 파악하기 힘들고 벡터가 너무 sparse하여 unique한 단어가 증가할 수록 필요로 하는 계산량이 매우 증가하게 됩니다.

2000년대에 이러한 단점을 개선한 NNLM이 등장합니다. 
NNLM은 어떤 단어 이전의 단어 n개를 one-hot encoding으로 vectorize하여 학습합니다. 
각각 벡터들을 Projection Layer를 통해 Hidden Layer로 전달되고 
이는 Output Layer에서 각 단어들이 나올 확률을 계산하여 실제 단어의 벡터와 비교하여 에러율을 계산하여 weight를 수정해 나갑니다. 

![NNLM](./NNLM.png)

이는 기본적으로 [Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_Hypothesis)을 기반으로 이루어 집니다. 비슷한 분포를 가지는 단어가 비슷한 의미를 가질 것 이므로, 학습된 NNLM은 비슷한 단어를 찾아내고 의미를 파악하는데 도움을 줄 수 있습니다.

NNLM은 단어의 벡터화라는 새로운 패러다임을 이끌었지만 몇 가지 단점이 있었습니다.

- 고정된  n값
- 단어 이후에 나오는 단어를 고려하지 못함.
- 학습 속도가 매우 느림

[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) [^2] 에서는 Word2vec을 소개하며 이러한 단점들을 극복하고 특히 학습 속도를 매우 개선하였습니다.

Word2vec은 CBOW(continuous bag-of-words)와 Skip-grams이라는 두 가지 모델 아키텍쳐를 제시하였습니다. 

CBOW모델은 주어진 단어 앞뒤의 단어를 Input으로 주어진 단어를 맞추기 위해 학습하는 네트워크를 구성하고, 
Skip-gram 모델은 단어 하나를 가지고 주위의 단어가 나올 확률을 추론합니다.

![CBOW and Skip-gram](./cbow-skip-gram.png)


여기에 Output Layer에서 Softmax를 계산할 때 Huffman Tree를 이용한 Hierarchical Softmax와 
Negative Sampling등의 방법을 이용해서 효과적으로 연산량을 줄였습니다. 

[genism](https://radimrehurek.com/gensim/models/word2vec.html)이나 [tensorflow](https://www.tensorflow.org/tutorials/word2vec)등, 구체화된 훌륭한 라이브러리들을 많이 찾을 수 있기 때문에 살펴보는 것을 권장합니다.

## 응용

Word2vec을 이용하여 단어를 vector space에 mapping할 수 있습니다. 
Vector space에 mapping된 vector들은 각 단어의 특성을 비슷하게 가지고 있습니다. 
각 벡터의 차이는 실제 단어의 의미상 차이와 유사해 지고, 비슷한 벡터는 의미적으로 비슷할 확률이 높습니다. 
예를 들어 `한국 – 서울 + 도쿄`라는 질의에 대해 `일본`을 반환해 줄 것입니다.

### 데이터 수집

우선 모델을 학습할 데이터를 모으는게 우선입니다. 
주변에서 쉽게 찾을 수 있는 풍부한 한국어 데이터로는 나무위키, 한국어 위키피디아, 한국어 공개 코퍼스 등이 있습니다.

혹은 책이나 뉴스 기사, 잡지, 연설문 등으로 해도 재밌는 결과가 나오며 실제 텍스트를 분석하는데 도움이 됩니다.
~~심지어 코드도!!~~

### 전처리

한국어의 경우 형태소 분석기를 통해 텍스트를 형태소 단위로 나누어 줘야 합니다. 띄어쓰기 단위로 단어만 분리하면 정확도가 높지 않습니다.
[KoNLPy](http://konlpy.org/ko/latest/)나 [은전한닢](http://eunjeon.blogspot.kr/) 같은 훌륭한 라이브러리들이 많이 있어 어렵지 않게 사용할 수 있습니다.
불용어도 제거해 주는 것이 성능을 향상시키는 데에 도움이 됩니다. 

이 부분은 어떤 형태소 분석기를 사용하더라도 완벽하게 형태소가 분리되기 어렵고,
형태소가 잘 나뉘어져 있다고 해도 어순에 의해 별로 연관되지 않은 단어가 연관되게 나타날 수 도 있습니다.
때문에 휴리스틱한 부분이 대다수 여서 딱히 정해진 규칙이나, 일반적인 방법을 찾아내기 힘듭니다.
자신의 데이터에 맞는 처리 방법을 찾아내는 것이 가장 좋습니다.

나무위키나 위키피디아의 경우 redirect나 틀, 혹은 여러 포맷들을 수정해 주어야합니다.

### 학습시키기

gensim에서 제공해주는 word2vec모델을 사용하면 손쉽게 학습시킬 수 있습니다.
[여기](https://radimrehurek.com/gensim/models/word2vec.html)를 참고해 보세요.

### 데모 만들기

웹이나 앱으로 만들어진 모델 파일을 이용해 조금만 처리를 하면 손쉽게 만들어 볼 수 있습니다.
실제 만들어진 [Demo](http:/server2.memento.live:5000)를 구경해 보실 수 있습니다. 데모의 [소스코드](https://github.com/memento7/word2demo)도 공개되어 있습니다.

## [Project: memento](https://memento7.github.io/2017/word2vec/)

[^1]: Firth, J. R. 1957:11
[^2]: Mikolov, Tomas; et al. "Efficient Estimation of Word Representations in Vector Space". 16 Jan 2013