---
title: "GPT-3 API를 처음 써본 날"
description: "2021년 6월, GPT-3 API 공개 직후의 경험과 생각"
date: 2021-06-01
slug: /gpt3-transformer-revolution-2021
tags: [ai]
published: true
---

# GPT-3 API를 처음 써본 날

2021년 6월, OpenAI가 GPT-3 API를 공개했습니다. 회사 슬랙에 누군가 공유한 링크를 보고, 저도 바로 API 접근 신청을 했습니다. 며칠 후 승인 메일을 받았을 때의 설렘이 아직도 기억납니다.

## 처음 API를 호출했을 때

API 키를 받자마자 터미널을 열고 여러 가지를 테스트해봤습니다. 번역, 요약, 코드 생성, 질의응답 등 생각나는 대로 시도해봤습니다.

```python
import openai

openai.api_key = "sk-..."

response = openai.Completion.create(
    engine="davinci",
    prompt="Translate English to Korean:\n\nEnglish: Hello, how are you?\nKorean:",
    max_tokens=50,
    temperature=0.3
)

print(response.choices[0].text)
# 출력: 안녕하세요, 어떻게 지내세요?
```

솔직히 첫 결과를 보고 놀랐습니다. 단순한 API 호출 몇 줄로 이런 품질의 번역이 나온다는 것이 믿기 어려웠습니다.

## Few-shot Learning의 신비

가장 인상 깊었던 것은 few-shot learning 능력이었습니다. 별도의 학습 과정 없이, 프롬프트에 예시 몇 개만 보여주면 패턴을 파악하고 따라하는 것을 직접 확인했습니다.

```python
prompt = """
프로그래밍 언어와 그 창시자:
Python: Guido van Rossum
Java: James Gosling
C++: Bjarne Stroustrup
Ruby:"""

response = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    max_tokens=20,
    temperature=0
)

print(response.choices[0].text.strip())
# 출력: Yukihiro Matsumoto
```

세 가지 예시만 보여줬는데, Ruby의 창시자를 정확하게 답했습니다. 이 정보가 학습 데이터에 있었겠지만, 패턴만 보고 적절한 형식으로 답변한다는 것이 신기했습니다. 마치 모델이 "아, 프로그래밍 언어와 창시자를 매핑하는 문제구나"라고 이해한 것처럼 느껴졌습니다.

더 복잡한 예제도 시도해봤습니다:

```python
prompt = """
다음 문장의 감정을 분석해주세요.

문장: 오늘 정말 기분이 좋아요!
감정: 긍정

문장: 이 영화는 시간 낭비였어요.
감정: 부정

문장: 내일 날씨가 어떨까요?
감정: 중립

문장: 이 제품 때문에 너무 화가 나요.
감정:"""

response = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    max_tokens=10,
    temperature=0
)

print(response.choices[0].text.strip())
# 출력: 부정
```

감정 분석 모델을 별도로 학습시키지 않아도, 예시 세 개만으로 새로운 문장의 감정을 분류할 수 있었습니다.

## 회사에서의 반응

당시 팀 내 반응은 두 갈래로 나뉘었습니다.

한쪽에서는 "이거 우리 일자리 없어지는 거 아니야?"라는 우려가 있었습니다. 특히 NLP 관련 업무를 하던 동료들이 신경을 많이 썼습니다. 모델 학습 없이 API 호출만으로 상당한 수준의 결과가 나오니, 자신들의 역할이 축소될 것 같다는 불안감이었습니다.

반대쪽에서는 "아직 프로덕션에 쓰기엔 무리"라는 의견이 있었습니다. 결과가 일관적이지 않고, 가끔 완전히 틀린 답을 자신있게 내놓는다는 점이 문제였습니다.

저는 중간쯤에 있었습니다. 분명 대단한 기술이지만, 당장 모든 것을 대체할 수준은 아니라고 봤습니다.

## 실제 업무에 적용해본 결과

몇 가지 실제 태스크에 GPT-3를 적용해봤습니다.

### 시도 1: 고객 문의 분류

```python
def classify_inquiry(text):
    prompt = f"""다음 고객 문의를 카테고리로 분류해주세요.
카테고리: 결제, 배송, 환불, 제품문의, 기타

문의: {text}
카테고리:"""

    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=10,
        temperature=0
    )
    return response.choices[0].text.strip()

# 테스트
print(classify_inquiry("주문한 상품이 언제 도착하나요?"))
# 출력: 배송
```

간단한 분류는 꽤 잘 작동했습니다. 하지만 문제가 있었습니다:

1. **일관성 부족**: 같은 문의에 대해 가끔 다른 답변이 나왔습니다.
2. **비용**: 문의량이 많으면 API 비용이 상당했습니다.
3. **속도**: 실시간 서비스에 넣기엔 latency가 길었습니다.

### 시도 2: 코드 생성

```python
prompt = """
Python 함수를 작성해주세요.
함수명: calculate_fibonacci
입력: n (정수)
출력: n번째 피보나치 수

함수:
```python
"""

response = openai.Completion.create(
    engine="davinci-codex",
    prompt=prompt,
    max_tokens=200,
    temperature=0
)
```

코드 생성은 인상적이었습니다. 특히 Codex 모델은 프로그래밍 관련 작업에서 높은 품질을 보여줬습니다. 다만 생성된 코드를 그대로 쓰기보다는 참고용으로 활용하는 것이 적절했습니다.

## 기술적으로 놀라웠던 점

175B(1,750억) 파라미터라는 규모 자체가 충격이었습니다. GPT-2가 1.5B였으니 100배 넘게 커진 것입니다. 단순히 크기만 키웠을 뿐인데 이전에 없던 능력들이 나타났습니다.

```python
# GPT-2 vs GPT-3 파라미터 비교
gpt2_params = 1.5e9   # 1.5B
gpt3_params = 175e9   # 175B

print(f"GPT-3는 GPT-2보다 {gpt3_params / gpt2_params:.0f}배 큽니다")
# 출력: GPT-3는 GPT-2보다 117배 큽니다
```

이런 현상을 **Emergent Abilities(창발적 능력)**라고 부른다는 것을 나중에 알게 되었습니다. 작은 모델에서는 전혀 보이지 않던 능력이 모델이 충분히 커지면 갑자기 나타난다는 개념입니다. 정확히 왜 이런 현상이 발생하는지는 아직도 완전히 규명되지 않았습니다.

## 당시의 예측과 실제

그때 몇 가지 예측을 했었는데, 맞은 것도 있고 틀린 것도 있습니다.

### 맞은 예측

- "앞으로 몇 년 안에 엄청난 발전이 있을 것" → 2022년 ChatGPT, 2023년 GPT-4로 현실이 되었습니다.
- "프롬프트 엔지니어링이 중요해질 것" → 실제로 프롬프트를 어떻게 작성하느냐에 따라 결과 품질이 크게 달라졌습니다.

### 틀린 예측

- "API 비용이 내려가면 다들 OpenAI API를 쓰겠지" → 오픈소스 모델들이 빠르게 따라잡았습니다. LLaMA가 나온 이후로는 직접 모델을 운영하는 것이 더 합리적인 경우도 많아졌습니다.
- "Hallucination 문제가 금방 해결될 것" → 2024년 현재까지도 완전히 해결되지 않았습니다.

## 개인적인 영향

GPT-3 이후로 AI에 대한 관심의 폭이 넓어졌습니다. 그전까지는 주로 컴퓨터 비전 쪽만 보고 있었는데, NLP 분야도 본격적으로 공부하기 시작했습니다.

```python
# 당시 공부하면서 작성한 Transformer 구현 일부
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        x = self.scaled_dot_product_attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(x)
```

지금 하고 있는 STT(Speech-to-Text) 업무도 결국 이때 생긴 관심이 계기가 되었습니다. 음성과 언어 모델을 연결하는 분야로 방향을 잡게 된 것은 GPT-3의 영향이 컸습니다.

## 돌아보며

2021년 6월의 그 순간이 일종의 전환점이었다고 생각합니다. "AI가 정말 뭔가 다르게 될 수 있겠구나"라는 느낌을 처음 받았던 때였습니다. 물론 그때는 이렇게까지 빨리 발전할 줄은 예상하지 못했습니다.

GPT-3를 처음 봤을 때 "대단하다"고 생각했는데, 지금 돌아보면 그것은 정말 시작에 불과했습니다. ChatGPT, GPT-4, 그리고 현재의 수많은 LLM들을 보면, 2021년의 GPT-3는 새로운 시대의 서막이었던 것 같습니다.

## 참고 자료

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3 논문
- [OpenAI API Documentation](https://platform.openai.com/docs) - API 문서
- [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258) - Stanford의 Foundation Model 리포트
