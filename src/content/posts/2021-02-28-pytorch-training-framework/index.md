---
title: "PyTorch 학습 프레임워크를 직접 만들어본 이야기"
description: "석사 연구하면서 매번 반복되는 코드에 지쳐서 만든 프레임워크"
date: 2021-02-28
permalink: /pytorch-training-framework
tags: [ai, dev]
published: true
---

# PyTorch 학습 프레임워크를 직접 만들어본 이야기

석사 과정 중에 모션 캡처 데이터로 페이셜 애니메이션 연구를 진행했습니다. 연구 자체보다 환경 세팅에 시간이 더 많이 들어서, 결국 학습 프레임워크를 직접 만들게 되었습니다. 이 글에서는 그 과정에서 배운 것들을 공유하고자 합니다.

## 왜 만들었나

매번 새로운 실험을 시작할 때마다 같은 코드를 복사해서 붙여넣는 것이 비효율적이라고 느꼈습니다. 데이터 로딩, 학습 루프, 체크포인트 저장, 로깅 등 기본적인 구성요소들을 매번 새로 작성하는 것은 시간 낭비였습니다.

특히 모션 캡처 데이터는 전처리 과정이 까다롭습니다. 결측치 처리, 노이즈 필터링, 시퀀스 길이 맞추기 등 다양한 처리가 필요합니다. 이런 작업들을 한 번 잘 만들어두면 이후 실험에서 재사용할 수 있겠다는 생각이 들었습니다.

```python
# 매번 이런 코드를 반복 작성하는 것이 비효율적이었습니다
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # 검증, 로깅, 체크포인트 저장 등...
```

## 프레임워크 구조

크게 세 가지 모듈로 구성했습니다.

### 데이터 파이프라인

모션 캡처 데이터를 효율적으로 처리하기 위한 파이프라인을 구축했습니다.

```python
class MotionDataset(Dataset):
    def __init__(self, data_path, sequence_length=120, overlap=60):
        self.data = self._load_motion_data(data_path)
        self.sequences = self._create_sequences(sequence_length, overlap)

    def _load_motion_data(self, path):
        """BVH/FBX 파일에서 모션 데이터 로드 및 전처리"""
        raw_data = load_motion_file(path)
        # 결측치 보간
        interpolated = self._interpolate_missing_frames(raw_data)
        # 노이즈 필터링 (Butterworth filter)
        filtered = butter_lowpass_filter(interpolated, cutoff=5, fs=60)
        return filtered

    def _create_sequences(self, length, overlap):
        """오버랩을 고려한 시퀀스 생성"""
        sequences = []
        stride = length - overlap
        for i in range(0, len(self.data) - length, stride):
            sequences.append(self.data[i:i+length])
        return sequences
```

주요 기능:
- **모션 캡처 데이터 로딩**: BVH, FBX 등 다양한 포맷 지원
- **결측치 처리**: Linear interpolation과 Kalman filter를 활용한 보간
- **시퀀스 생성**: 학습을 위한 오버랩 시퀀스 생성
- **동적 배치 사이징**: 메모리 사용량에 따른 배치 크기 자동 조절

### 학습 루프 (Trainer)

기본적인 학습/검증 루프를 추상화하여 재사용 가능하도록 구성했습니다.

```python
class Trainer:
    def __init__(self, model, optimizer, criterion, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.best_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch['input'].to(self.device))
            loss = self.criterion(outputs, batch['target'].to(self.device))

            # Backward pass
            loss.backward()

            # Gradient clipping (모션 데이터는 값 범위가 커서 필수)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def check_early_stopping(self, val_loss):
        """조기 종료 체크"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            self.save_checkpoint('best_model.pt')
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience
```

Trainer가 제공하는 기능:
- **기본 train/eval 루프**: 표준적인 학습/검증 사이클
- **조기 종료 (Early Stopping)**: Validation loss 기반 자동 중단
- **학습률 스케줄링**: Warmup, Cosine Annealing 등 지원
- **체크포인트 자동 저장**: Best model 및 주기적 저장
- **Gradient Clipping**: 모션 데이터 특성상 gradient 폭발 방지

### 설정 관리 (Config)

실험별 설정을 YAML 파일로 관리하여 재현 가능한 실험 환경을 구축했습니다.

```yaml
# configs/facial_animation_v1.yaml
experiment:
  name: "facial_motion_transformer"
  seed: 42

model:
  type: "transformer"
  d_model: 256
  nhead: 8
  num_encoder_layers: 6
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 200
  patience: 20
  grad_clip: 1.0

data:
  sequence_length: 120
  overlap: 60
  train_ratio: 0.8
```

```python
from dataclasses import dataclass
from omegaconf import OmegaConf

@dataclass
class Config:
    experiment: dict
    model: dict
    training: dict
    data: dict

    @classmethod
    def from_yaml(cls, path):
        return OmegaConf.load(path)

    def save(self, path):
        OmegaConf.save(self, path)

# 사용 예시
config = Config.from_yaml('configs/facial_animation_v1.yaml')
set_seed(config.experiment.seed)  # 재현 가능성 보장
```

## 실험 관리와 로깅

실험 결과를 추적하고 비교하기 위해 [Weights & Biases](https://wandb.ai/)를 통합했습니다.

```python
import wandb

class ExperimentLogger:
    def __init__(self, config, project_name="motion-research"):
        self.run = wandb.init(
            project=project_name,
            config=OmegaConf.to_container(config),
            name=config.experiment.name
        )

    def log_metrics(self, metrics, step):
        wandb.log(metrics, step=step)

    def log_animation(self, motion_data, name):
        """생성된 모션을 시각화하여 로깅"""
        video = render_motion_to_video(motion_data)
        wandb.log({name: wandb.Video(video, fps=30)})
```

## 실제 효과

프레임워크를 구축한 후 확실히 연구 효율이 향상되었습니다.

| 항목 | 이전 | 이후 | 개선율 |
|------|------|------|--------|
| 실험 설정 시간 | 2.5시간 | 1시간 | 60% 단축 |
| 새 모델 테스트 | 하루 | 반나절 | 50% 단축 |
| 버그 발생률 | 높음 | 낮음 | - |

숫자로 보면 60% 정도 단축되었는데, 체감상으로는 그 이상이었습니다. 매번 "이전에 어떻게 했더라?"하며 옛날 코드를 찾아보는 시간이 없어지고, 실험 재현이 YAML 파일 하나로 가능해졌기 때문입니다.

## 아쉬운 점과 배운 것

솔직히 지금 돌아보면 부끄러운 부분도 있습니다. 당시에는 [PyTorch Lightning](https://lightning.ai/)이나 [Hugging Face Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) 같은 훌륭한 도구들이 이미 존재했는데, 그런 것들을 충분히 알아보지 않고 직접 만들었습니다.

만약 그때 이런 도구들을 알았다면 굳이 직접 만들지 않았을 것 같습니다. 하지만 직접 만들면서 얻은 것도 있습니다:

1. **데이터 로딩 최적화에 대한 이해**: `DataLoader`의 `num_workers`, `pin_memory`, `prefetch_factor` 등의 옵션이 성능에 미치는 영향을 직접 체감했습니다.

2. **메모리 관리**: Gradient accumulation, mixed precision training 등을 직접 구현하면서 GPU 메모리 관리에 대해 깊이 이해하게 되었습니다.

3. **디버깅 능력**: 학습이 잘 안 될 때 어디서 문제가 생기는지 파악하는 능력이 향상되었습니다.

```python
# 직접 구현하면서 배운 메모리 최적화 기법
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 현재는

지금은 PyTorch Lightning을 주로 사용합니다. 제가 만든 것보다 훨씬 완성도가 높고, 커뮤니티 지원도 활발합니다. 하지만 당시에는 직접 만든 프레임워크가 필요했고, 석사 연구를 끝내는 데 큰 도움이 되었기 때문에 후회는 없습니다.

## 교훈

이 경험에서 얻은 교훈을 정리하면 다음과 같습니다:

1. **표준 도구부터 찾아보자**: 바퀴를 재발명하기 전에 이미 존재하는 솔루션이 있는지 충분히 조사해야 합니다.

2. **그래도 직접 만들면 배우긴 한다**: 삽질도 학습입니다. 추상화된 도구를 쓰기만 하면 내부 동작을 이해하기 어렵습니다.

3. **자동화에 투자하는 건 남는 장사**: 처음에는 시간이 들어도 나중에 충분히 회수됩니다.

4. **설정 관리와 재현성이 중요하다**: 실험 결과를 신뢰하려면 동일한 설정으로 동일한 결과가 나와야 합니다.

마지막으로, 연구나 프로젝트를 진행할 때 "이 작업을 앞으로 몇 번이나 반복하게 될까?"를 생각해보시는 것을 권장합니다. 두세 번 이상이라면 자동화나 추상화에 투자하는 것이 결국 더 효율적입니다.
