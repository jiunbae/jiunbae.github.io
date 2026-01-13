---
title: "SDXL로 텍스처 생성 프로젝트를 진행하면서"
description: "Stable Diffusion XL을 실제 게임 에셋 파이프라인에 적용해본 경험"
date: 2023-04-01
slug: /stable-diffusion-generative-ai-2023
tags: [ai, dev]
published: true
---

# SDXL로 텍스처 생성 프로젝트를 진행하면서

2023년 하반기, Stable Diffusion XL(SDXL)이 공개되었을 때 솔직히 충격을 받았습니다. 이전 버전들은 "AI가 그린 거 티가 난다"는 느낌이 있었는데, SDXL은 품질이 한 단계 올라갔습니다. 회사에서 텍스처 생성 프로젝트를 진행하면서 SDXL을 본격적으로 활용해봤고, 그 경험을 공유합니다.

## SDXL의 달라진 점

SDXL은 기존 SD 1.5나 SD 2.1과 비교해서 몇 가지 중요한 개선이 있었습니다.

```python
from diffusers import StableDiffusionXLPipeline
import torch

# SDXL 로드
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe = pipe.to("cuda")

# 기본 해상도가 1024x1024
image = pipe(
    prompt="a medieval castle, detailed architecture, golden hour lighting",
    height=1024,
    width=1024,
    num_inference_steps=40,
    guidance_scale=7.5
).images[0]
```

주요 개선점:
- **해상도**: 기본 1024x1024 (기존 512x512)
- **디테일**: 훨씬 선명하고 세밀한 표현
- **손/얼굴**: 여전히 완벽하지는 않지만 많이 개선됨
- **프롬프트 이해**: 복잡한 프롬프트도 더 잘 따름

## 텍스처 생성 프로젝트

회사에서 3D 에셋의 텍스처를 빠르게 프로토타이핑하는 프로젝트를 진행했습니다. 기존에는 아티스트가 수작업으로 그리던 것을, AI로 초안을 만들고 수정하는 방식으로 파이프라인을 개선하려 했습니다.

### 워크플로우 설계

```
[기존 파이프라인]
1. 아티스트가 레퍼런스 수집 (2-3일)
2. 컨셉 스케치 (1-2일)
3. 텍스처 제작 (3-5일)
4. 피드백 및 수정 (1-2일)
총: 7-12일

[AI 보조 파이프라인]
1. SDXL로 컨셉 이미지 다량 생성 (몇 시간)
2. 아티스트가 선별 및 방향 결정 (반나절)
3. ControlNet으로 구조 잡고 텍스처 생성 (1일)
4. 아티스트가 후처리 및 완성 (2-3일)
총: 3-5일
```

시간을 약 50% 단축할 수 있었습니다.

### ControlNet과의 결합

핵심은 [ControlNet](https://github.com/lllyasviel/ControlNet)과 함께 사용하는 것이었습니다. 아티스트가 기본 스케치를 그리면, 그 구조를 유지하면서 텍스처를 입히는 방식입니다.

```python
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from PIL import Image
import torch

# ControlNet 로드 (Canny edge용)
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 스케치 이미지 로드 및 edge 추출
def get_canny_edge(image_path):
    import cv2
    import numpy as np

    image = cv2.imread(image_path)
    image = cv2.Canny(image, 100, 200)
    image = Image.fromarray(image)
    return image

sketch = get_canny_edge("artist_sketch.png")

# 스케치 구조를 유지하면서 텍스처 생성
result = pipe(
    prompt="stone castle wall texture, medieval, weathered, moss",
    image=sketch,
    controlnet_conditioning_scale=0.8,  # 스케치 영향력 조절
    num_inference_steps=30
).images[0]
```

여러 종류의 ControlNet을 상황에 따라 사용했습니다:

```python
CONTROLNET_CONFIGS = {
    "structure": {
        # 구조/형태를 잡을 때
        "model": "diffusers/controlnet-depth-sdxl-1.0",
        "scale": 0.7
    },
    "outline": {
        # 윤곽선 기반 생성
        "model": "diffusers/controlnet-canny-sdxl-1.0",
        "scale": 0.8
    },
    "normal": {
        # 노멀맵 기반 (3D 텍스처에 유용)
        "model": "controlnet-normal-sdxl",
        "scale": 0.6
    }
}
```

### 커스텀 LoRA 학습

프로젝트 아트 스타일에 맞는 결과물을 얻기 위해 커스텀 LoRA를 학습시켰습니다.

```python
# LoRA 학습 설정 (kohya_ss 스크립트 기반)
training_config = {
    "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
    "train_data_dir": "./training_images",  # 30-50장의 레퍼런스 이미지
    "output_dir": "./lora_output",

    # LoRA 파라미터
    "network_dim": 32,  # rank
    "network_alpha": 16,

    # 학습 파라미터
    "learning_rate": 1e-4,
    "max_train_epochs": 10,
    "save_every_n_epochs": 2,

    # 메모리 최적화
    "mixed_precision": "fp16",
    "gradient_checkpointing": True,
}
```

학습 과정에서 배운 점들:

1. **데이터 품질이 중요**: 30장의 좋은 이미지가 100장의 평범한 이미지보다 낫습니다.
2. **캡션이 중요**: 각 이미지에 대한 설명을 잘 달아야 합니다.
3. **과적합 주의**: 에포크를 너무 많이 돌리면 과적합됩니다.

```python
# 학습된 LoRA 적용
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)

# LoRA 로드
pipe.load_lora_weights("./lora_output", weight_name="project_style.safetensors")

# LoRA 가중치 조절 (0.0 ~ 1.0)
pipe.fuse_lora(lora_scale=0.8)
```

## 실제 작업 결과

### 잘 된 것들

```python
# 컨셉 아트 초안 - 매우 효과적
good_use_cases = [
    "배경 컨셉 아트",
    "환경 텍스처 (돌, 나무, 금속 등)",
    "프롭 디자인 초안",
    "무드보드 / 레퍼런스 이미지"
]

# 예시 프롬프트
prompts = {
    "stone_wall": "seamless stone wall texture, medieval castle, weathered, mossy, 4k, detailed",
    "wood_floor": "wooden floor planks, aged oak, scratched, warm lighting, seamless texture",
    "metal_plate": "rusted metal plate, industrial, scratches and dents, seamless, pbr"
}
```

### 아직 어려운 것들

```python
# 한계가 있는 경우들
challenging_cases = [
    "정확한 글씨/로고 - 여전히 이상하게 나옴",
    "일관된 캐릭터 - 같은 캐릭터 여러 포즈 어려움",
    "정확한 구도/레이아웃 - 대략적으로만 가능",
    "완성품 수준 퀄리티 - 항상 후처리 필요"
]
```

## 아티스트와의 협업

"AI가 아티스트를 대체하는 거 아니야?"라는 우려가 처음에 있었습니다. 하지만 실제로 써보니 오히려 아티스트의 역할이 더 중요해졌습니다.

```
AI가 잘하는 것:
- 대량의 옵션 빠르게 생성
- 레퍼런스 이미지 만들기
- 초안/프로토타입

아티스트가 잘하는 것:
- "이건 된다/안 된다" 판단
- 방향 설정과 큐레이션
- 세밀한 수정과 완성
- 프로젝트 스타일 일관성 유지
```

결국 AI는 도구이고, 그 도구를 어떻게 활용하느냐는 사람의 역할입니다.

## 비용 분석

```python
# 로컬 GPU vs 클라우드 API 비용 비교

local_gpu_cost = {
    "hardware": "RTX 4090 (약 250만원)",
    "전기세": "월 약 3-5만원",
    "생성량": "하루 수천 장 가능",
    "장기_비용": "6개월 이상 사용 시 유리"
}

cloud_api_cost = {
    "per_image": "약 $0.01-0.02",
    "daily_1000": "약 $10-20",
    "monthly": "약 $300-600 (활발히 사용 시)",
    "장점": "초기 투자 없음, 관리 불필요"
}
```

우리 프로젝트에서는 생성량이 많아서 로컬 GPU가 유리했습니다. A100 한 대로 팀 전체가 사용했습니다.

## 저작권 고려사항

상업 프로젝트에서 가장 신경 쓴 부분입니다.

```python
# 저작권 리스크 최소화 전략
safety_guidelines = {
    "avoid": [
        "특정 아티스트 이름 사용",
        "특정 작품 언급",
        "유명 IP 관련 프롬프트"
    ],
    "prefer": [
        "일반적인 스타일 설명 (예: 'watercolor style')",
        "자체 데이터로 학습한 LoRA 사용",
        "결과물 후처리로 변형"
    ]
}
```

우리 프로젝트에서는 자체 레퍼런스 이미지로 LoRA를 학습시켜서 리스크를 줄였습니다.

## 2023년을 돌아보며

SDXL은 "이미지 생성 AI가 드디어 실무에서 쓸만해졌다"를 보여준 모델이었습니다. 완벽하지는 않지만, 워크플로우에 잘 통합하면 생산성을 크게 높일 수 있었습니다.

2024년에는 더 발전하리라 예상합니다. 특히 기대하는 것들:
- 일관된 캐릭터 생성 (IP-Adapter 등의 발전)
- 비디오 생성 (Stable Video Diffusion의 발전)
- 더 빠른 생성 속도 (SDXL Turbo, Lightning 등)

> **2024년 업데이트**: 예상대로 많은 발전이 있었습니다.

```python
POST_SDXL_DEVELOPMENTS = {
    "SD3 (2024.02)": {
        "architecture": "DiT (Diffusion Transformer) 기반",
        "highlight": "텍스트 렌더링 크게 개선",
        "note": "SDXL과 다른 아키텍처로 전환"
    },
    "FLUX (2024.08)": {
        "creator": "Black Forest Labs",
        "highlight": "현재 오픈소스 이미지 생성 최고 품질",
        "variants": ["FLUX.1 [dev]", "FLUX.1 [schnell]"]
    },
    "video_generation": {
        "Sora": "OpenAI (비공개)",
        "Runway Gen-3": "상업 서비스",
        "CogVideo": "오픈소스 대안"
    }
}
```

## 관련 글

- [GPT-3가 바꿔놓은 것들](/posts/gpt3-transformer-innovation)
- [Stable Diffusion을 실무에 적용해본 경험](/posts/stable-diffusion-ai-generation-2023)
- [LLaMA가 바꿔놓은 것들](/posts/llama-opensource-ecosystem-2023)

---

## 참고 자료

- [Stability AI - SDXL](https://stability.ai/stable-diffusion)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [kohya-ss LoRA Training](https://github.com/kohya-ss/sd-scripts)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 노드 기반 워크플로우 도구
