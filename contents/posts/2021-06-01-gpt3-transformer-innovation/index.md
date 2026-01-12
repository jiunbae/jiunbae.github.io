---
title: "GPT-3가 바꿔놓은 것들"
description: "2020년 GPT-3 발표 이후 AI 업계가 어떻게 달라졌는지"
date: 2021-06-01
slug: /gpt3-transformer-innovation
tags: [ai]
published: true
---

# GPT-3가 바꿔놓은 것들

2020년 6월, OpenAI가 GPT-3를 공개했습니다. 1,750억 개의 파라미터를 가진 거대 언어 모델이었습니다. 숫자만 들으면 "그래서 뭐가 다른 건데?"라고 생각할 수 있지만, 이 모델이 AI 연구와 산업에 미친 영향은 생각보다 훨씬 컸습니다.

## 스케일링이 답이었다

GPT-3 이전의 AI 연구는 "어떤 구조가 더 좋을까"를 고민하는 것이 대부분이었습니다. LSTM이 좋은지 GRU가 좋은지, Attention을 어디에 붙일지, 어떤 정규화 기법을 쓸지 등 모델 아키텍처에 대한 논의가 주를 이루었습니다.

그런데 GPT-3는 다른 메시지를 던졌습니다.

> "구조는 그냥 Transformer 쓰고, 대신 크기를 키워라."

실제로 GPT-3는 GPT-2와 구조적으로 크게 다르지 않습니다. 동일한 Transformer decoder 아키텍처를 사용하며, 주요 차이점은 규모입니다:

| 모델 | 파라미터 수 | 학습 데이터 |
|------|-----------|------------|
| GPT-2 | 1.5B | 40GB |
| GPT-3 | 175B | 570GB |

100배 이상 큰 모델이 100배 이상의 데이터로 학습되었을 뿐인데, 이 단순한 접근이 예상치 못한 능력들을 발현시켰습니다.

## In-Context Learning과 Emergent Abilities

GPT-3에서 가장 인상적이었던 것은 **few-shot learning** 능력입니다. 모델을 별도로 fine-tuning하지 않아도, 프롬프트에 예시를 몇 개 보여주면 새로운 태스크를 수행할 수 있습니다.

```python
# Few-shot learning 예시
prompt = """
Translate English to French:
English: Hello, how are you?
French: Bonjour, comment allez-vous?

English: What is your name?
French: Comment vous appelez-vous?

English: I love programming.
French:
"""

# GPT-3 응답: "J'aime la programmation."
```

프로그래밍 언어 창시자 예시도 흥미로웠습니다:

```
Python: Guido van Rossum
Java: James Gosling
Rust: ???
```

이런 패턴만 보여주면 GPT-3는 "Graydon Hoare"라고 답합니다. 별도 학습 없이 패턴을 파악하고 적용하는 것입니다.

이러한 능력을 **Emergent Ability(창발적 능력)**라고 부릅니다. 모델이 충분히 커지면 명시적으로 가르치지 않은 능력이 나타난다는 개념입니다. 정확히 왜 이런 현상이 발생하는지는 아직 완전히 규명되지 않았지만, 스케일링의 힘을 보여주는 인상적인 사례였습니다.

## 연구 패러다임의 변화

GPT-3 이후 AI 연구의 트렌드가 크게 바뀌었습니다.

### 과거: 아이디어 중심

예전에는 작은 팀이 좋은 아이디어만 있으면 State-of-the-Art를 달성할 수 있었습니다. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 논문 하나가 전체 판을 뒤집은 것처럼요. 대학 연구실에서도 충분히 경쟁력 있는 연구가 가능했습니다.

### 현재: 자원 중심

GPT-3급 모델을 학습시키려면:
- **GPU**: 수천 개의 A100 GPU
- **비용**: 수백만 달러의 학습 비용
- **데이터**: 수백 GB 이상의 고품질 텍스트 데이터
- **시간**: 수개월의 학습 시간

학교 연구실에서 이 규모의 모델을 학습시키는 것은 현실적으로 불가능합니다.

```python
# GPT-3 학습에 필요한 대략적인 계산량
# (이론적 추정)
params = 175e9  # 175B parameters
tokens = 300e9  # 300B tokens
flops_per_token = 6 * params  # forward + backward
total_flops = flops_per_token * tokens

# 약 3.14e23 FLOPS
# V100 GPU로 약 355 GPU-년 필요
```

이 변화가 좋은 것인지 나쁜 것인지는 아직 판단하기 어렵습니다. 빅테크만 연구를 주도하게 되는 것 같아 씁쓸한 면이 있지만, 어쨌든 기술 발전은 가속화되고 있습니다.

## 실무에서의 변화

GPT-3는 실무에도 직접적인 영향을 미쳤습니다. 당시 제 업무에서 느낀 변화들입니다:

### 1. API로 해결 가능한 영역 확대

```python
# 2021년 당시 코드 (현재는 deprecated)
# import openai
# response = openai.Completion.create(engine="text-davinci-003", ...)

# 현재 OpenAI SDK (v1.0+) 방식
from openai import OpenAI

client = OpenAI()

def classify_sentiment(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 또는 gpt-3.5-turbo
        messages=[
            {"role": "system", "content": "Classify the sentiment as positive, negative, or neutral."},
            {"role": "user", "content": text}
        ],
        max_tokens=10
    )
    return response.choices[0].message.content.strip()

# 기존에는 모델 학습이 필요했던 작업이 몇 줄로 해결
```

텍스트 분류, 요약, 번역 같은 간단한 NLP 태스크가 API 호출 몇 줄로 해결 가능해졌습니다.

### 2. "AI 엔지니어" 역할 변화

AI 엔지니어의 역할이 "모델 학습하는 사람"에서 "프롬프트 잘 짜는 사람"으로 일부 재정의되기 시작했습니다. 물론 이것은 일부 영역에 한정된 이야기이지만, 분명한 변화였습니다.

### 3. 새로운 집중 영역

API로 해결되는 문제가 늘어나면서 "그럼 우리가 뭘 해야 하지?"라는 고민이 생겼습니다. 결론은 API로 안 되는 것들에 집중하는 것이었습니다:

- **도메인 특화**: 특정 분야의 전문 지식이 필요한 태스크
- **Latency 최적화**: 실시간 응답이 필요한 서비스
- **프라이버시**: 민감한 데이터를 외부 API로 보낼 수 없는 경우
- **비용 효율**: 대량 처리 시 API 비용이 부담되는 경우

## 한계와 문제점

물론 GPT-3가 만능은 아니었습니다.

### Hallucination (환각)

```
Q: 파리는 어느 나라의 수도인가요?
A: 파리는 독일의 수도입니다.  # 틀린 답을 자신있게 말함
```

GPT-3는 자신있게 틀린 정보를 생성하는 경우가 많았습니다. 이 문제는 현재의 LLM들도 완전히 해결하지 못했습니다.

### 일관성 부재

```python
# 같은 질문에 다른 답변
response1 = gpt3("2 + 2 = ?")  # "4"
response2 = gpt3("2 + 2 = ?")  # "The answer is 4."
response3 = gpt3("2 + 2 = ?")  # "2 plus 2 equals four."
```

Temperature 설정에 따라 같은 질문에도 다른 형식의 답변이 나왔습니다. 프로덕션에서 일관된 결과가 필요할 때 문제가 됐습니다.

### 비용

2021년 당시 GPT-3 API 비용은 상당했습니다:
- Davinci (가장 성능 좋은 모델): $0.06 / 1K tokens
- 대량 처리 시 비용이 빠르게 증가

지금은 많이 저렴해졌지만, 당시에는 실험하는 것조차 부담스러웠습니다.

## Scaling Law의 발견

GPT-3와 함께 중요한 논문이 발표되었습니다: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

이 논문에서 제시한 핵심 발견:

```
Loss ∝ N^(-α) · D^(-β) · C^(-γ)

N: 모델 파라미터 수
D: 학습 데이터 크기
C: 학습에 사용된 연산량
```

모델 크기, 데이터, 연산량을 늘리면 **예측 가능한 방식으로** 성능이 향상된다는 것입니다. 이는 앞으로 더 큰 모델을 만들면 어느 정도의 성능 향상을 기대할 수 있는지 예측할 수 있게 해주었습니다.

> **후속 연구 (2022년 Chinchilla)**: DeepMind의 Chinchilla 논문은 GPT-3가 "compute-optimal"하지 않았음을 보여주었습니다. 동일한 연산 예산이라면 모델 크기와 데이터 크기를 균형 있게 늘리는 것이 더 효율적이라는 것입니다. 이후 LLaMA 같은 모델들은 이 교훈을 반영하여 더 적은 파라미터로도 높은 성능을 달성했습니다.

## 지금 돌아보면

2021년의 GPT-3는 시작점이었습니다. 2022년 ChatGPT, 2023년 GPT-4로 이어지면서 당시의 예감이 맞았다는 것이 확인되었습니다.

그때 "이거 뭔가 다르다"라고 느꼈던 것이 틀리지 않았습니다. 다만 이렇게까지 빨리 발전할 줄은 예상하지 못했습니다. 2021년에 GPT-3를 보면서 "대단하다"라고 생각했는데, 지금 돌아보면 그것은 정말 시작에 불과했습니다.

결국 중요한 것은 기술 자체보다 그것을 어떻게 활용하느냐인 것 같습니다. GPT-3가 나왔을 때 "이걸 어떻게 쓸 수 있을까"를 빨리 고민한 사람들이 지금 앞서 나가고 있으니까요.

## 참고 자료

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3 논문
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) - 스케일링 법칙 논문
- [OpenAI API Documentation](https://platform.openai.com/docs) - GPT-3 API 문서
