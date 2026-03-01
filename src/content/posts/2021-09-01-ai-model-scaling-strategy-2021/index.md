---
title: "2021년 AI 모델 스케일링 전략과 경험"
description: "GPT-3 이후 대규모 모델 학습에 대해 배운 것들"
date: 2021-09-01
permalink: /ai-model-scaling-strategy-2021
tags: [ai, dev]
published: true
---

# 2021년 AI 모델 스케일링 전략과 경험

2021년은 AI 분야에서 "더 크게"가 화두였습니다. GPT-3가 175B 파라미터로 인상적인 성능을 보여준 이후, 모델 크기 경쟁이 본격화되었습니다. 저도 회사에서 10B 규모의 모델을 학습시켜본 경험이 있어서, 그 과정에서 배운 것들을 정리해보았습니다.

## 스케일링의 세 가지 축

대규모 모델을 학습시키기 위한 분산 학습 방법은 크게 세 가지로 나눌 수 있습니다.

### 1. 데이터 병렬화 (Data Parallelism)

가장 기본적이고 널리 쓰이는 방법입니다. 동일한 모델을 여러 GPU에 복제하고, 데이터를 나눠서 처리합니다.

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def train_with_ddp(rank, world_size):
    setup(rank, world_size)

    # 모델을 현재 GPU로 이동
    model = MyModel().to(rank)

    # DDP로 래핑
    model = DDP(model, device_ids=[rank])

    # 데이터 로더 (각 GPU가 다른 데이터를 받음)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, sampler=sampler, batch_size=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 셔플링을 위해 필요
        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()

# 실행
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train_with_ddp, args=(world_size,), nprocs=world_size)
```

PyTorch의 `DistributedDataParallel`(DDP)을 사용하면 비교적 쉽게 구현할 수 있습니다. 각 GPU가 gradient를 계산하고, all-reduce 연산으로 gradient를 평균 내어 동기화합니다.

**장점**: 구현이 간단하고 효율적입니다.
**단점**: 모델 전체가 각 GPU에 올라가야 하므로, GPU 메모리보다 큰 모델은 불가능합니다.

### 2. 모델 병렬화 (Model Parallelism)

모델 자체를 여러 GPU에 나눠서 배치하는 방법입니다. 하나의 GPU에 올라가지 않는 거대 모델을 학습시킬 때 필수입니다.

```python
import torch
import torch.nn as nn

class ModelParallelTransformer(nn.Module):
    def __init__(self, num_layers=24, d_model=1024):
        super().__init__()

        # 레이어를 두 GPU에 나눔
        self.embedding = nn.Embedding(50000, d_model).to('cuda:0')

        # 전반부 레이어 → GPU 0
        self.layers_gpu0 = nn.ModuleList([
            TransformerLayer(d_model) for _ in range(num_layers // 2)
        ]).to('cuda:0')

        # 후반부 레이어 → GPU 1
        self.layers_gpu1 = nn.ModuleList([
            TransformerLayer(d_model) for _ in range(num_layers // 2)
        ]).to('cuda:1')

        self.output = nn.Linear(d_model, 50000).to('cuda:1')

    def forward(self, x):
        # GPU 0에서 처리
        x = x.to('cuda:0')
        x = self.embedding(x)
        for layer in self.layers_gpu0:
            x = layer(x)

        # GPU 1로 이동 후 처리
        x = x.to('cuda:1')
        for layer in self.layers_gpu1:
            x = layer(x)

        return self.output(x)
```

**장점**: 단일 GPU 메모리보다 큰 모델을 학습할 수 있습니다.
**단점**: GPU 간 통신 오버헤드가 크고, GPU 활용률이 떨어집니다.

### 3. 파이프라인 병렬화 (Pipeline Parallelism)

모델 병렬화의 GPU 활용률 문제를 개선한 방법입니다. 배치를 마이크로 배치로 나누고, 파이프라인처럼 처리합니다.

```python
# 개념적인 파이프라인 병렬화 설명
"""
기존 모델 병렬화:
Time →  [GPU0: Layer 1-12]  [GPU1: Layer 13-24]
Batch 1:    ████████████████    ████████████████
           └─ GPU1은 GPU0이 끝날 때까지 대기 (낭비)

파이프라인 병렬화:
Time →
GPU0:  [μB1][μB2][μB3][μB4]  ← 마이크로 배치들
GPU1:       [μB1][μB2][μB3][μB4]
           └─ 이전 마이크로 배치 처리와 병렬 실행
"""

# PyTorch의 pipeline 예시 (간략화)
from torch.distributed.pipeline.sync import Pipe

model = nn.Sequential(
    nn.Linear(1024, 1024),  # Layer 1
    nn.ReLU(),
    nn.Linear(1024, 1024),  # Layer 2
    # ... 더 많은 레이어
)

# 2개 GPU에 파이프라인으로 분할
model = Pipe(model, chunks=8)  # 8개 마이크로 배치로 분할
```

**장점**: GPU 활용률이 모델 병렬화보다 높습니다.
**단점**: 구현이 복잡하고, bubble (idle time)이 여전히 존재합니다.

## DeepSpeed와 ZeRO

2021년에 분산 학습의 판도를 바꾼 것은 Microsoft의 [DeepSpeed](https://www.deepspeed.ai/)였습니다. 특히 ZeRO(Zero Redundancy Optimizer) 기술이 인상적이었습니다.

### ZeRO의 핵심 아이디어

기존 데이터 병렬화에서는 각 GPU가 모델 전체의 복사본을 가지고 있습니다. 여기에는 세 가지가 포함됩니다:
- **모델 파라미터 (Parameters)**
- **그래디언트 (Gradients)**
- **옵티마이저 상태 (Optimizer States)**

Adam 옵티마이저의 경우, 옵티마이저 상태가 파라미터의 2배 메모리를 차지합니다. 즉, FP16 모델 기준으로:

```
메모리 사용량 (GPU당):
- 파라미터: 2 bytes × N
- 그래디언트: 2 bytes × N
- 옵티마이저 상태 (Adam): 4 bytes × N + 4 bytes × N = 8 bytes × N
총: 12 bytes × N (파라미터 수)

10B 모델 기준:
10B × 12 bytes = 120GB per GPU
→ A100 80GB로도 불가능
```

ZeRO는 이 상태들을 GPU 간에 분산하여 중복을 제거합니다:

```python
# DeepSpeed ZeRO Stage 3 사용 예시
import deepspeed
import torch

# DeepSpeed 설정 파일 (ds_config.json)
ds_config = {
    "train_batch_size": 256,
    "gradient_accumulation_steps": 16,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,  # ZeRO Stage 3: 파라미터, 그래디언트, 옵티마이저 상태 모두 분산
        "offload_optimizer": {
            "device": "cpu",  # 옵티마이저 상태를 CPU로 오프로드
        },
        "offload_param": {
            "device": "cpu",  # 파라미터도 CPU로 오프로드 가능
        }
    }
}

# 모델 초기화
model = MyLargeModel()

# DeepSpeed로 래핑
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=ds_config
)

# 학습 루프
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

ZeRO Stage 3를 사용하면 메모리 효율이 크게 향상됩니다:

```
ZeRO Stage 3 메모리 사용량 (N개 GPU):
- 파라미터: 2 bytes × N / N = 2 bytes × N (통신 시에만 수집)
- 그래디언트: 2 bytes × N / N
- 옵티마이저 상태: 8 bytes × N / N

10B 모델, 8 GPU 기준:
(2 + 2 + 8) × 10B / 8 = 15GB per GPU
→ A100 80GB로 충분히 가능
```

## 직접 10B 모델을 학습시켜 본 경험

회사에서 10B 규모의 모델을 학습시켜봤습니다. A100 8대로 약 2주가 걸렸습니다.

### 환경 구성

```bash
# NVIDIA 컨테이너 사용
docker pull nvcr.io/nvidia/pytorch:21.08-py3

# 노드 간 통신을 위한 NCCL 설정
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# DeepSpeed 실행
deepspeed --num_gpus=8 \
    --hostfile=hostfile \
    train.py \
    --deepspeed_config ds_config.json
```

### 겪었던 문제들

**1. 분산 학습 디버깅의 어려움**

분산 학습에서 버그가 발생하면 디버깅이 매우 어렵습니다. 어떤 노드에서 문제가 생겼는지 파악하기 힘들고, 재현도 어렵습니다.

```python
# 분산 환경에서의 디버깅 팁
import torch.distributed as dist

def debug_log(message):
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[Rank {rank}] {message}")

# 체크포인트를 자주 저장
def save_checkpoint(model, optimizer, epoch, loss, path):
    if dist.get_rank() == 0:  # 마스터 노드에서만 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)
    dist.barrier()  # 모든 노드가 저장 완료를 기다림
```

**2. OOM (Out of Memory) 문제**

처음에는 배치 사이즈를 너무 크게 잡아서 OOM이 자주 발생했습니다. Gradient accumulation으로 해결했습니다.

```python
# Gradient Accumulation 예시
accumulation_steps = 16
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**3. 학습 불안정성**

대규모 모델은 학습이 불안정해지기 쉽습니다. Loss가 갑자기 발산하는 경우가 있었습니다.

```python
# 학습 안정화를 위한 기법들

# 1. Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Learning Rate Warmup
def get_lr(step, warmup_steps=1000, max_lr=1e-4):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    return max_lr

# 3. Loss Scaling (FP16 학습 시)
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

with autocast():
    loss = model(batch)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### 체크포인트 관리의 중요성

10B 모델의 체크포인트는 용량이 매우 큽니다. 체크포인트 관리 전략이 중요했습니다.

```python
# 체크포인트 저장 전략
class CheckpointManager:
    def __init__(self, save_dir, max_checkpoints=3):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def save(self, model, optimizer, epoch, loss):
        path = f"{self.save_dir}/checkpoint_epoch{epoch}.pt"

        # 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)

        self.checkpoints.append(path)

        # 오래된 체크포인트 삭제 (최근 N개만 유지)
        while len(self.checkpoints) > self.max_checkpoints:
            old_path = self.checkpoints.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)
```

## 혼합 정밀도 학습 (Mixed Precision Training)

FP16 학습은 2021년에 거의 표준이 되었습니다. 메모리를 절반으로 줄이면서 속도도 빨라집니다.

```python
from torch.cuda.amp import autocast, GradScaler

# Mixed Precision Training 기본 패턴
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # autocast 블록 내에서 연산은 자동으로 FP16으로 수행
    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    # Gradient scaling으로 언더플로우 방지
    scaler.scale(loss).backward()

    # Gradient unscale 후 클리핑
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
```

주의할 점:
- **Loss Scaling**: FP16의 좁은 표현 범위로 인한 gradient 언더플로우를 방지합니다.
- **특정 연산은 FP32 유지**: LayerNorm, Softmax 등은 FP32로 계산해야 수치 안정성이 보장됩니다.

## 2021년의 교훈

### 효율성이 우선이다

무작정 자원을 늘리는 것보다 최적화가 먼저입니다. DeepSpeed의 ZeRO, Gradient checkpointing, Mixed precision 등의 기법으로 같은 하드웨어에서 훨씬 큰 모델을 학습시킬 수 있습니다.

```python
# Gradient Checkpointing 예시
from torch.utils.checkpoint import checkpoint

class EfficientTransformerLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = MultiHeadAttention(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        # checkpoint로 메모리 절약 (연산은 재계산하지만 메모리는 절약)
        x = x + checkpoint(self.attention, x)
        x = x + checkpoint(self.ffn, x)
        return x
```

### 빅테크와의 격차

솔직히, GPU 수천 개를 돌릴 수 있는 곳과 그렇지 않은 곳의 격차가 느껴졌습니다. GPT-3급 모델을 처음부터 학습시키는 것은 대부분의 조직에서 불가능합니다.

하지만 오픈소스 모델들이 나오면서 이 격차가 조금씩 줄고 있습니다. 직접 학습시키지 않아도 공개된 모델을 fine-tuning하여 활용할 수 있게 되었습니다.

### 인프라 엔지니어링의 중요성

대규모 모델 학습에서는 ML 알고리즘만큼이나 인프라 엔지니어링이 중요합니다. 분산 시스템, 네트워크 최적화, 스토리지 관리 등의 역량이 필요합니다.

```bash
# 학습 모니터링 스크립트 예시
#!/bin/bash

# GPU 상태 모니터링
watch -n 1 nvidia-smi

# 노드 간 네트워크 대역폭 확인
iperf3 -c other_node -p 5001

# 학습 로그 실시간 확인
tail -f logs/training.log | grep -E "(loss|step|lr)"
```

## 마치며

2021년은 스케일링의 해였습니다. "모델을 크게 만들면 성능이 좋아진다"는 단순한 원리가 실제로 동작한다는 것이 확인되었고, 이를 가능하게 하는 분산 학습 기술들이 빠르게 발전했습니다.

직접 대규모 모델을 학습시켜본 경험은 소중했습니다. 논문에서 읽는 것과 실제로 해보는 것은 완전히 다른 차원의 이해를 제공합니다. 분산 학습의 어려움, 최적화의 중요성, 그리고 인프라의 가치를 직접 체감할 수 있었습니다.

## 참고 자료

- [DeepSpeed Documentation](https://www.deepspeed.ai/docs/)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)
