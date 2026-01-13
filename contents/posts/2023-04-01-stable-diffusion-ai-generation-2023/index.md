---
title: "Stable Diffusion을 실무에 적용해본 경험"
description: "2023년 생성형 이미지 AI를 업무에 활용하면서 배운 것들"
date: 2023-04-01
slug: /stable-diffusion-ai-generation-2023
tags: [ai]
published: true
---

# Stable Diffusion을 실무에 적용해본 경험

2023년 초, Stable Diffusion을 처음 사용해봤습니다. "텍스트를 입력하면 이미지가 생성된다"는 개념은 알고 있었지만, 직접 써보니 그 가능성과 한계가 동시에 느껴졌습니다. 이 글에서는 실제 업무에 적용하면서 배운 것들을 공유합니다.

## 첫 만남

[Hugging Face의 diffusers](https://huggingface.co/docs/diffusers) 라이브러리를 사용하면 몇 줄의 코드로 이미지를 생성할 수 있습니다.

```python
from diffusers import StableDiffusionPipeline
import torch

# 모델 로드
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 이미지 생성
prompt = "a cat sitting on a desk, digital art, highly detailed"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("cat.png")
```

처음 생성된 이미지를 봤을 때, 솔직히 놀랐습니다. 몇 줄의 코드로 이 정도 품질의 이미지가 나온다는 것이 신기했습니다.

물론 완벽하지는 않았습니다. 손가락이 6개라거나, 눈이 이상하게 그려지는 경우도 있었습니다. 하지만 "대충 이런 느낌"을 빠르게 시각화하는 용도로는 충분했습니다.

## 실무 활용 사례

### 1. 프레젠테이션 삽화

회사에서 발표 자료를 만들 때, 삽화가 필요한 경우가 많습니다. 기존에는 스톡 이미지 사이트를 뒤지거나 직접 그려야 했는데, Stable Diffusion으로 빠르게 생성할 수 있게 되었습니다.

```python
def generate_presentation_image(concept, style="minimalist"):
    prompt = f"{concept}, {style} style, clean background, vector art"
    negative_prompt = "text, watermark, signature, blurry"

    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.0
    ).images[0]

    return image

# 사용 예시
image = generate_presentation_image(
    "cloud server architecture diagram",
    style="flat design"
)
```

스톡 이미지를 검색하는 것보다 원하는 이미지를 바로 생성하는 것이 더 빨랐습니다. 특히 추상적인 개념(예: "AI와 인간의 협업")을 시각화할 때 유용했습니다.

### 2. 블로그 썸네일

기술 블로그 포스트에 썸네일이 필요할 때도 활용했습니다.

```python
def generate_blog_thumbnail(topic, size=(1200, 630)):
    # OG 이미지 권장 크기
    prompt = f"abstract representation of {topic}, modern, tech, gradient colors"

    image = pipe(
        prompt,
        width=size[0],
        height=size[1],
        num_inference_steps=40
    ).images[0]

    return image
```

### 3. 아이디어 스케치

새로운 기능이나 UI를 구상할 때, 아이디어를 빠르게 시각화하는 용도로 사용했습니다.

```python
# 앱 UI 컨셉 스케치
prompts = [
    "mobile app dashboard, dark mode, data visualization, modern UI",
    "chat interface, messaging app, clean design, minimalist",
    "settings page, toggle switches, iOS style"
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt, num_inference_steps=30).images[0]
    image.save(f"concept_{i}.png")
```

실제 디자인 작업을 시작하기 전에 여러 방향을 탐색하는 데 유용했습니다.

## LoRA: 스타일 커스터마이징

기본 모델로 원하는 스타일이 나오지 않을 때, [LoRA (Low-Rank Adaptation)](https://huggingface.co/docs/diffusers/training/lora)를 활용했습니다.

```python
from diffusers import StableDiffusionPipeline
import torch

# 기본 모델 로드
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# LoRA 가중치 로드 (커뮤니티에서 공유된 스타일)
pipe.load_lora_weights("path/to/lora/weights")

# LoRA가 적용된 상태로 생성
image = pipe(
    "a landscape, illustration style",
    num_inference_steps=40
).images[0]
```

커뮤니티에서 공유하는 LoRA를 다운받아 적용하면, 특정 화풍이나 스타일로 일관된 이미지를 생성할 수 있었습니다. 일러스트 스타일, 애니메이션 스타일, 사진 스타일 등 다양한 옵션이 있었습니다.

직접 LoRA를 학습시키는 것도 시도해봤습니다.

```python
# LoRA 학습 (간략화된 예시)
from diffusers import DiffusionPipeline
from peft import LoraConfig, get_peft_model

# LoRA 설정
lora_config = LoraConfig(
    r=4,  # rank
    lora_alpha=32,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    lora_dropout=0.1,
)

# 레퍼런스 이미지 30~50장으로 학습
# (실제 학습 코드는 더 복잡합니다)
```

A100 GPU로 2~3시간 정도면 커스텀 LoRA를 학습시킬 수 있었습니다.

## 프롬프트 엔지니어링

원하는 결과를 얻기 위해서는 프롬프트 작성이 중요했습니다. 시행착오를 통해 몇 가지 패턴을 발견했습니다.

```python
# 효과적인 프롬프트 구조
def build_prompt(subject, style, quality_tags, negative_tags):
    """
    좋은 프롬프트 구조:
    [주제] + [스타일] + [품질 태그] + [기술적 디테일]
    """
    prompt = f"{subject}, {style}, {', '.join(quality_tags)}"

    negative_prompt = ", ".join(negative_tags)

    return prompt, negative_prompt

# 사용 예시
prompt, negative = build_prompt(
    subject="a futuristic city skyline at sunset",
    style="cyberpunk, neon lights",
    quality_tags=["highly detailed", "8k", "artstation", "concept art"],
    negative_tags=["blurry", "low quality", "text", "watermark", "signature"]
)
```

자주 사용한 프롬프트 패턴:

```python
PROMPT_TEMPLATES = {
    "concept_art": "{subject}, concept art, highly detailed, artstation, digital painting",
    "photo_realistic": "{subject}, photorealistic, 8k, professional photography, detailed",
    "minimalist": "{subject}, minimalist, clean, vector, simple shapes, flat design",
    "illustration": "{subject}, illustration, digital art, vibrant colors, stylized",
}

NEGATIVE_PROMPTS = {
    "default": "blurry, low quality, distorted, deformed, ugly, duplicate",
    "no_text": "text, words, letters, watermark, signature, logo",
    "realistic": "cartoon, anime, illustration, drawing, painting",
}
```

## 한계와 주의점

### 1. 텍스트 렌더링의 한계

Stable Diffusion은 텍스트를 잘 그리지 못합니다. "Happy Birthday"라고 쓰라고 하면 이상한 글자가 나옵니다.

```python
# 이렇게 하면 안 됨
bad_prompt = "a birthday card with text 'Happy Birthday'"

# 대신 이렇게
good_prompt = "a birthday card, festive, colorful, celebration theme"
# → 텍스트는 후처리로 추가
```

텍스트가 필요한 이미지는 생성 후 Pillow나 Photoshop으로 텍스트를 따로 추가해야 했습니다.

### 2. 일관성 문제

같은 캐릭터를 여러 포즈로 그리기가 어렵습니다. 프롬프트를 똑같이 해도 매번 다른 캐릭터가 나옵니다.

```python
# 시드 고정으로 어느 정도 완화 가능
generator = torch.Generator("cuda").manual_seed(42)

image1 = pipe("a red-haired girl", generator=generator).images[0]
image2 = pipe("a red-haired girl, different pose", generator=generator).images[0]
# → 그래도 완전히 같은 캐릭터는 어려움
```

### 3. 저작권 이슈

학습 데이터에 무엇이 포함되어 있는지 정확히 알 수 없습니다. 특정 아티스트의 스타일로 생성하면 저작권 문제가 있을 수 있습니다.

```python
# 주의가 필요한 프롬프트
risky_prompt = "in the style of [specific artist name]"

# 더 안전한 대안
safer_prompt = "digital art style, vibrant colors"
```

상업적으로 사용할 때는 조심해야 합니다. 저는 외부 배포용에는 사용을 자제하고, 내부 아이디어 스케치나 레퍼런스 용도로만 활용했습니다.

## 로컬 환경 구성

API 비용을 아끼고 프라이버시를 보호하기 위해 로컬에서 돌리는 것이 좋습니다.

```python
# 메모리 최적화 설정
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,  # 메모리 절약 (주의해서 사용)
)

# Attention Slicing (메모리 절약)
pipe.enable_attention_slicing()

# xformers 사용 (속도 향상)
pipe.enable_xformers_memory_efficient_attention()

pipe = pipe.to("cuda")
```

RTX 3090 (24GB VRAM) 기준으로:
- 512x512 이미지: 약 3초
- 768x768 이미지: 약 5초
- 1024x1024 이미지: 약 10초

하루에 수백 장 생성해도 전기세만 나갑니다. API를 사용하는 것보다 훨씬 경제적이었습니다.

## 결론

Stable Diffusion은 "도구"로서 충분히 유용합니다. 디자이너를 대체하는 것은 아니지만, 아이디어를 빠르게 시각화하고 레퍼런스를 만드는 데 탁월합니다.

다만 한계도 명확합니다:
- 텍스트 렌더링 불가
- 일관된 캐릭터 생성 어려움
- 저작권 이슈

이런 한계를 인식하고 적절한 용도로 활용한다면, 생산성을 크게 높일 수 있는 도구라고 생각합니다.

## 후속 발전 (2024년 이후)

> **2024년 업데이트**: 이 글 작성 이후 이미지 생성 AI는 빠르게 발전했습니다.

```python
STABLE_DIFFUSION_EVOLUTION = {
    "SDXL (2023.07)": {
        "resolution": "1024x1024 기본",
        "improvement": "품질 대폭 향상, 텍스트 렌더링 약간 개선",
        "note": "이 글에서 다룬 SD 1.5보다 훨씬 좋은 품질"
    },
    "SDXL Turbo (2023.11)": {
        "speed": "1-4 step으로 생성 가능",
        "improvement": "실시간 생성에 가까운 속도"
    },
    "SD3 (2024.02)": {
        "architecture": "DiT (Diffusion Transformer) 기반",
        "improvement": "텍스트 렌더링 크게 개선, 프롬프트 이해력 향상",
        "note": "드디어 글씨가 제대로 나옴"
    },
    "FLUX (2024.08)": {
        "creator": "Black Forest Labs (SD 창시자들)",
        "improvement": "SD3보다 더 나은 품질, 빠른 속도",
        "note": "오픈소스 이미지 생성의 새로운 기준"
    }
}
```

특히 SD3와 FLUX는 이 글에서 언급한 "텍스트 렌더링 불가" 문제를 상당 부분 해결했습니다. 기술 발전 속도가 정말 빠릅니다.

## 관련 글

- [GPT-3가 바꿔놓은 것들](/posts/gpt3-transformer-innovation)
- [SDXL로 텍스처 생성 프로젝트를 진행하면서](/posts/stable-diffusion-generative-ai-2023)
- [LLaMA가 바꿔놓은 것들](/posts/llama-opensouce-ecosystem-2023)

## 참고 자료

- [Stable Diffusion 공식 문서](https://stability.ai/stable-diffusion)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)
- [Civitai](https://civitai.com/) - LoRA 모델 공유 커뮤니티
- [AUTOMATIC1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - GUI 기반 사용
