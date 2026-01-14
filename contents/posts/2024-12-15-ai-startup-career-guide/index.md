---
title: "AI 스타트업 연구 엔지니어로 일한다는 것"
description: "대기업에서 스타트업으로 이직하면서 느낀 점들과 실무 경험, 기술 스택, 커리어 가이드"
date: 2024-12-15
slug: /ai-startup-career-guide
tags: [ai, career, startup, mlops]
published: true
---

# AI 스타트업 연구 엔지니어로 일한다는 것

NCSOFT에서 3년, Return Zero에서 현재까지. 대기업과 스타트업 양쪽을 경험하면서 느낀 점들을 정리해봅니다. AI 분야에서 커리어를 시작하거나, 이직을 고민하는 분들께 도움이 되었으면 합니다.

## 왜 스타트업으로 갔나

솔직히 말하면 NCSOFT가 싫어서는 아니었습니다. 오히려 좋은 환경이었습니다. 데이터도 충분했고, 동료들도 훌륭했고, 연구할 시간도 주어졌습니다.

```python
# 대기업의 장점 (개인적 경험)
NCSOFT_PROS = {
    "data": {
        "scale": "수십 TB의 게임 데이터",
        "quality": "정제된 고품질 데이터",
        "accessibility": "내부 데이터 플랫폼 잘 구축됨"
    },
    "colleagues": {
        "expertise": "각 분야 전문가들",
        "learning": "시니어에게 배울 기회 많음",
        "collaboration": "체계적인 협업 프로세스"
    },
    "resources": {
        "compute": "A100 클러스터 자유롭게 사용",
        "time": "장기 프로젝트 가능",
        "support": "인프라팀, 데이터팀 지원"
    }
}
```

그런데 한 가지가 답답했습니다. 제가 만든 것이 실제 서비스에 들어가기까지 너무 오래 걸렸습니다.

```
[대기업에서의 전형적인 프로세스]

1. 연구 아이디어 제안 (2주)
   └─ 상위 보고, 승인 대기

2. 프로토타입 개발 (1-2개월)
   └─ 내부 데모, 성능 검증

3. 기술 검토 (1개월)
   └─ 보안 검토, 법무 검토, 인프라 검토

4. 개발팀 전달 (2주)
   └─ 문서화, 인수인계, 코드 리뷰

5. 서비스 통합 (2-3개월)
   └─ 개발팀에서 재구현, 최적화

6. QA 및 배포 (1개월)
   └─ 테스트, 단계별 배포

총 소요: 6-8개월
```

연구 결과가 실제 서비스에 반영되기까지 반년이 넘게 걸리는 경우가 많았습니다. 그 사이에 기술 트렌드가 바뀌기도 하고, 더 좋은 방법이 나오기도 합니다. "내가 만든 게 실제로 쓰이는 걸 보고 싶다"는 욕구가 커졌습니다.

스타트업은 다릅니다.

```
[스타트업에서의 프로세스]

1. 아이디어 → 구현 → 배포
   └─ 같은 날 또는 며칠 내

2. 피드백 수집 → 개선 → 재배포
   └─ 매일 또는 매주

3. 반복
```

물론 그만큼 정신없지만, 내가 만든 것이 바로 돌아가는 것을 볼 수 있다는 건 큰 동기부여가 됩니다.

### 이직 결심의 결정적 계기

구체적으로 말하자면, 제가 개발한 모델이 서비스에 반영되기까지 8개월이 걸렸는데, 그 사이에 GPT-4가 나와서 접근 방식 자체를 바꿔야 했던 경험이 있었습니다. 대기업의 안정성은 좋지만, AI 분야의 빠른 변화 속도와는 맞지 않는다는 생각이 들었습니다.

```python
# 이직 결심 체크리스트 (당시 제 생각)
DECISION_FACTORS = {
    "push_factors": {  # 떠나게 만든 요인
        "slow_deployment": "연구 → 서비스 배포까지 6-8개월",
        "indirect_impact": "내가 만든 게 어디에 쓰이는지 모름",
        "bureaucracy": "작은 변경도 승인 절차 필요",
        "comfort_zone": "익숙한 환경에서 성장 정체 느낌"
    },
    "pull_factors": {  # 끌어당긴 요인
        "direct_impact": "내가 만든 게 바로 서비스됨",
        "ownership": "프로젝트 전체를 책임짐",
        "learning_speed": "다양한 기술 경험 가능",
        "equity": "성공 시 큰 보상 (스톡옵션)"
    },
    "risk_assessment": {
        "financial": "6개월 생활비 비상금 확보",
        "career": "실패해도 배운 경험은 남음",
        "age": "리스크 감수 가능한 나이 (30대 초반)"
    }
}
```

## 연구 엔지니어라는 직군

요즘 "연구 엔지니어(Research Engineer)" 또는 "ML 엔지니어"라는 타이틀이 조금 애매해졌습니다. 회사마다, 팀마다 기대하는 바가 다릅니다.

```python
# 과거 vs 현재의 역할 변화
ROLE_EVOLUTION = {
    "2018-2020": {
        "title": "ML Research Engineer",
        "primary": "논문 구현, 모델 학습",
        "secondary": "프로토타입 개발",
        "rarely": "서빙, 인프라",
        "tools": ["PyTorch", "TensorFlow", "Jupyter"]
    },
    "2024": {
        "title": "ML/AI Engineer",
        "primary": "모델 개발 + 서빙",
        "secondary": "파이프라인 구축",
        "often": "프로덕션 최적화, 모니터링",
        "tools": [
            "PyTorch", "Transformers",  # 모델링
            "FastAPI", "Triton",  # 서빙
            "Docker", "K8s",  # 인프라
            "Prometheus", "Grafana"  # 모니터링
        ]
    }
}
```

예전에는 논문 쓰고 모델 만드는 사람이었다면, 지금은 그것에 더해서 서빙도 하고, 파이프라인도 짜고, 가끔 프론트엔드까지 건드립니다. 특히 스타트업에서는 "그거 니가 해"가 자주 나옵니다.

처음에는 "저 연구원인데..."라고 생각했는데, 하다 보니 그게 오히려 재미있습니다. 내가 만든 모델이 어떻게 서비스되는지 직접 보면서 개선점을 찾을 수 있으니까요.

### 직군별 상세 비교

현재 AI/ML 분야에서 흔히 볼 수 있는 직군들을 정리해봤습니다.

```python
# AI/ML 관련 직군 비교
ML_ROLES_COMPARISON = {
    "ML Research Scientist": {
        "focus": "새로운 알고리즘/아키텍처 연구",
        "output": "논문, 특허",
        "coding_ratio": "30-50%",
        "requirements": ["PhD 선호", "논문 실적", "수학적 기반"],
        "companies": ["Google Brain", "OpenAI", "DeepMind"],
        "korean_companies": ["NAVER AI Lab", "Kakao Brain", "삼성리서치"]
    },
    "ML Engineer": {
        "focus": "모델 학습 및 서빙 파이프라인",
        "output": "프로덕션 모델, 인프라",
        "coding_ratio": "70-90%",
        "requirements": ["CS/ML 기초", "엔지니어링 역량", "시스템 이해"],
        "companies": ["Meta", "Netflix", "Uber"],
        "korean_companies": ["쿠팡", "당근", "토스"]
    },
    "Research Engineer": {
        "focus": "연구 아이디어의 구현 및 검증",
        "output": "프로토타입, 실험 결과",
        "coding_ratio": "60-80%",
        "requirements": ["논문 구현 능력", "실험 설계", "빠른 학습"],
        "companies": ["Anthropic", "Cohere", "Hugging Face"],
        "korean_companies": ["리턴제로", "업스테이지", "포티투닷"]
    },
    "MLOps Engineer": {
        "focus": "ML 시스템 운영 및 자동화",
        "output": "파이프라인, 모니터링 시스템",
        "coding_ratio": "70-90%",
        "requirements": ["DevOps 경험", "분산 시스템", "클라우드"],
        "companies": ["Spotify", "Airbnb", "LinkedIn"],
        "korean_companies": ["라인", "카카오", "배달의민족"]
    }
}
```

### "풀스택 ML"의 현실

스타트업에서는 하루에도 여러 가지 역할을 오가게 됩니다.

```python
# 실제 하루 업무 예시
FULLSTACK_ML_REALITY = {
    "morning": {
        "task": "모델 학습 결과 확인",
        "tools": ["wandb", "tensorboard"],
        "code_example": """
# 아침에 확인하는 코드
import wandb

api = wandb.Api()
runs = api.runs("my-project/stt-experiments")

# 밤새 돌린 실험 결과 확인
for run in runs[:5]:
    if run.state == "finished":
        print(f"{run.name}: CER={run.summary.get('cer', 'N/A'):.4f}")
        """
    },
    "afternoon": {
        "task": "서빙 API 버그 수정",
        "tools": ["FastAPI", "Triton", "Docker"],
        "code_example": """
# 발견된 버그: 특수문자 포함 입력 시 에러
@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    # 기존 코드: 바로 처리
    # audio_data = await audio.read()

    # 수정: 유효성 검사 추가
    audio_data = await audio.read()
    if len(audio_data) > MAX_AUDIO_SIZE:
        raise HTTPException(400, "Audio file too large")
    if not is_valid_audio(audio_data):
        raise HTTPException(400, "Invalid audio format")

    return await process_audio(audio_data)
        """
    },
    "evening": {
        "task": "학습 파이프라인 개선",
        "tools": ["Airflow", "Ray"],
        "code_example": """
# 데이터 전처리 병렬화로 학습 시작 시간 단축
from ray.data import read_json
import ray

@ray.remote
def preprocess_batch(batch):
    # 오디오 리샘플링, 노말라이즈, 특성 추출
    return [preprocess_audio(item) for item in batch]

# 기존: 순차 처리 (2시간)
# 개선: Ray로 병렬 처리 (15분)
futures = [preprocess_batch.remote(batch) for batch in batches]
results = ray.get(futures)
        """
    },
    "sometimes": {
        "task": "데모 페이지 만들기",
        "tools": ["Gradio", "Streamlit", "React"],
        "code_example": """
import gradio as gr

def transcribe_demo(audio):
    # 모델 추론
    result = model.transcribe(audio)
    return result["text"], result["confidence"]

demo = gr.Interface(
    fn=transcribe_demo,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=[
        gr.Textbox(label="인식 결과"),
        gr.Number(label="신뢰도")
    ],
    title="STT Demo",
    description="마이크로 녹음하면 텍스트로 변환합니다"
)

demo.launch(share=True)  # 외부 공유 링크 생성
        """
    }
}
```

## 실제로 하는 일

Return Zero에서 하는 일을 비율로 나누면 대략 이렇습니다.

```python
# 업무 비율 (개인적 경험)
WORK_DISTRIBUTION = {
    "model_research": {
        "percentage": 40,
        "activities": [
            "논문 읽고 구현",
            "모델 아키텍처 실험",
            "학습 코드 작성",
            "하이퍼파라미터 튜닝"
        ],
        "tools": ["PyTorch", "Transformers", "wandb"]
    },
    "mlops": {
        "percentage": 30,
        "activities": [
            "학습 파이프라인 구축",
            "데이터 파이프라인 관리",
            "모델 버전 관리",
            "실험 추적"
        ],
        "tools": ["Airflow", "MLflow", "DVC"]
    },
    "service_development": {
        "percentage": 20,
        "activities": [
            "API 개발",
            "성능 최적화",
            "모니터링 설정",
            "장애 대응"
        ],
        "tools": ["FastAPI", "Triton", "Prometheus"]
    },
    "others": {
        "percentage": 10,
        "activities": [
            "미팅",
            "문서화",
            "채용 인터뷰",
            "외부 발표"
        ]
    }
}
```

### 실제 프로젝트 예시: STT 모델 개발

제가 참여한 STT 프로젝트의 실제 업무 흐름을 공유합니다.

```python
# 프로젝트 타임라인 (약 3개월)
PROJECT_TIMELINE = {
    "week_1_2": {
        "phase": "리서치 & 기획",
        "tasks": [
            "기존 STT 솔루션 벤치마크",
            "요구사항 정의 (CER < 4%, latency < 200ms)",
            "모델 아키텍처 선정 (FastConformer 결정)"
        ],
        "output": "기술 검토 문서"
    },
    "week_3_4": {
        "phase": "데이터 준비",
        "tasks": [
            "학습 데이터 수집 (KsponSpeech, 자체 데이터)",
            "데이터 전처리 파이프라인 구축",
            "품질 검증 (샘플링 기반 수작업 검수)"
        ],
        "output": "정제된 데이터셋 500시간"
    },
    "week_5_8": {
        "phase": "모델 개발",
        "tasks": [
            "베이스라인 모델 학습",
            "하이퍼파라미터 튜닝",
            "한국어 특화 개선 (자모 토크나이저 등)"
        ],
        "output": "CER 3.1% 달성 모델"
    },
    "week_9_10": {
        "phase": "최적화 & 서빙",
        "tasks": [
            "ONNX 변환 및 TensorRT 최적화",
            "Triton 서버 설정",
            "로드 테스트"
        ],
        "output": "latency 150ms, 동시 100 요청 처리"
    },
    "week_11_12": {
        "phase": "배포 & 모니터링",
        "tasks": [
            "단계적 배포 (canary → full)",
            "모니터링 대시보드 구축",
            "알림 설정"
        ],
        "output": "프로덕션 배포 완료"
    }
}
```

실제 코드 레벨에서 어떤 작업을 하는지 보여드리겠습니다.

```python
# 1. 데이터 전처리 파이프라인
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class STTDataset(Dataset):
    def __init__(self, manifest_path: str, tokenizer, sample_rate: int = 16000):
        self.samples = self._load_manifest(manifest_path)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

        # 오디오 변환기
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=48000,  # 원본 샘플레이트 (가정)
            new_freq=sample_rate
        )

    def _load_manifest(self, path: str):
        """JSONL 형식의 manifest 파일 로드"""
        import json
        samples = []
        with open(path, 'r') as f:
            for line in f:
                samples.append(json.loads(line))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # 오디오 로드 및 전처리
        waveform, sr = torchaudio.load(sample["audio_path"])
        if sr != self.sample_rate:
            waveform = self.resampler(waveform)

        # 모노로 변환
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 텍스트 토크나이징
        tokens = self.tokenizer.encode(sample["text"])

        return {
            "audio": waveform.squeeze(0),
            "audio_length": waveform.shape[1],
            "tokens": torch.tensor(tokens),
            "token_length": len(tokens)
        }

def collate_fn(batch):
    """Dynamic padding을 위한 collate 함수"""
    # 오디오 패딩
    max_audio_len = max(item["audio_length"] for item in batch)
    audios = torch.zeros(len(batch), max_audio_len)
    audio_lengths = torch.zeros(len(batch), dtype=torch.long)

    # 토큰 패딩
    max_token_len = max(item["token_length"] for item in batch)
    tokens = torch.zeros(len(batch), max_token_len, dtype=torch.long)
    token_lengths = torch.zeros(len(batch), dtype=torch.long)

    for i, item in enumerate(batch):
        audios[i, :item["audio_length"]] = item["audio"]
        audio_lengths[i] = item["audio_length"]
        tokens[i, :item["token_length"]] = item["tokens"]
        token_lengths[i] = item["token_length"]

    return {
        "audio": audios,
        "audio_lengths": audio_lengths,
        "tokens": tokens,
        "token_lengths": token_lengths
    }
```

```python
# 2. 학습 루프 (간소화 버전)
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb

class STTTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps,
            T_mult=2
        )

        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        # wandb 초기화
        wandb.init(project="stt-training", config=vars(config))

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Forward
            logits = self.model(batch["audio"].cuda())

            # CTC Loss 계산
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)  # (T, N, C)

            loss = self.criterion(
                log_probs,
                batch["tokens"].cuda(),
                batch["audio_lengths"].cuda() // self.config.downsample_factor,
                batch["token_lengths"].cuda()
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            # 로깅
            if batch_idx % self.config.log_interval == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/step": epoch * len(self.train_loader) + batch_idx
                })

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_cer = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # 추론
                logits = self.model(batch["audio"].cuda())
                predictions = self.decode(logits)

                # CER 계산
                for pred, target in zip(predictions, batch["tokens"]):
                    cer = self.calculate_cer(pred, target)
                    total_cer += cer
                    total_samples += 1

        avg_cer = total_cer / total_samples
        wandb.log({"val/cer": avg_cer})
        return avg_cer
```

```python
# 3. 모델 서빙 코드
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
import tritonclient.grpc as grpcclient
import numpy as np
import io
import soundfile as sf

app = FastAPI(title="STT API")

# Triton 클라이언트 초기화
triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

class TranscriptionResponse(BaseModel):
    text: str
    confidence: float
    processing_time_ms: float

@app.post("/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe(audio: UploadFile):
    """음성 파일을 텍스트로 변환합니다."""
    import time
    start_time = time.time()

    # 파일 유효성 검사
    if not audio.content_type.startswith("audio/"):
        raise HTTPException(400, "Audio file required")

    # 오디오 로드
    audio_bytes = await audio.read()
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes))

    # 리샘플링 (16kHz로)
    if sample_rate != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)

    # Triton 요청 준비
    audio_input = grpcclient.InferInput("audio", waveform.shape, "FP32")
    audio_input.set_data_from_numpy(waveform.astype(np.float32))

    # 추론
    response = triton_client.infer(
        model_name="stt_model",
        inputs=[audio_input],
        outputs=[
            grpcclient.InferRequestedOutput("text"),
            grpcclient.InferRequestedOutput("confidence")
        ]
    )

    text = response.as_numpy("text")[0].decode("utf-8")
    confidence = float(response.as_numpy("confidence")[0])

    processing_time = (time.time() - start_time) * 1000

    return TranscriptionResponse(
        text=text,
        confidence=confidence,
        processing_time_ms=processing_time
    )

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    try:
        if triton_client.is_server_ready():
            return {"status": "healthy"}
        else:
            raise HTTPException(503, "Triton server not ready")
    except Exception as e:
        raise HTTPException(503, f"Health check failed: {str(e)}")
```

비율은 프로젝트 상황에 따라 많이 달라집니다.

```python
# 상황별 업무 비율 변화
WORK_BY_SITUATION = {
    "normal": {
        "research": 40,
        "mlops": 30,
        "service": 20,
        "others": 10
    },
    "launch_approaching": {
        "research": 10,
        "mlops": 20,
        "service": 60,  # 서비스 개발에 집중
        "others": 10
    },
    "post_launch_stabilization": {
        "research": 20,
        "mlops": 40,  # 운영 안정화
        "service": 30,
        "others": 10
    },
    "exploratory_phase": {
        "research": 60,  # 새로운 방향 탐색
        "mlops": 20,
        "service": 10,
        "others": 10
    }
}
```

## 기술 스택 상세

스타트업 ML 엔지니어로서 실제로 사용하는 기술 스택을 정리했습니다.

### 모델 개발

```python
# 모델 개발 스택
MODEL_DEVELOPMENT_STACK = {
    "deep_learning_frameworks": {
        "primary": "PyTorch 2.x",
        "why": "동적 그래프, 디버깅 용이, 커뮤니티",
        "alternatives": ["JAX (연구용)", "TensorFlow (레거시)"]
    },
    "high_level_apis": {
        "nlp": "Transformers (Hugging Face)",
        "speech": "NeMo (NVIDIA), SpeechBrain",
        "vision": "timm, torchvision"
    },
    "experiment_tracking": {
        "primary": "Weights & Biases",
        "why": "시각화 좋음, 팀 협업, 하이퍼파라미터 스윕",
        "alternatives": ["MLflow (self-hosted)", "Neptune"]
    },
    "distributed_training": {
        "data_parallel": "PyTorch DDP",
        "model_parallel": "DeepSpeed, FSDP",
        "cluster": "Ray Train"
    }
}
```

실제로 실험 설정 파일을 어떻게 관리하는지 보여드리겠습니다.

```yaml
# configs/stt_fastconformer.yaml
model:
  name: "FastConformer-CTC"
  encoder:
    d_model: 512
    n_layers: 18
    n_heads: 8
    conv_kernel_size: 31
    dropout: 0.1
  decoder:
    type: "ctc"
    vocab_size: 5000

data:
  train_manifest: "/data/train.jsonl"
  val_manifest: "/data/val.jsonl"
  sample_rate: 16000
  max_duration: 20.0  # 최대 20초 오디오

training:
  batch_size: 32
  accumulate_grad_batches: 4
  max_epochs: 100
  learning_rate: 1e-3
  warmup_steps: 5000
  weight_decay: 1e-6
  max_grad_norm: 1.0

  # Early stopping
  early_stopping:
    monitor: "val/cer"
    patience: 10
    mode: "min"

  # Checkpointing
  checkpoint:
    save_top_k: 3
    monitor: "val/cer"

augmentation:
  speed_perturb: [0.9, 1.0, 1.1]
  noise_aug:
    enabled: true
    snr_range: [10, 30]
  spec_augment:
    freq_masks: 2
    time_masks: 10
    freq_width: 27
    time_width: 0.05

wandb:
  project: "stt-fastconformer"
  tags: ["korean", "ctc", "production"]
```

```python
# 설정 파일 로드 및 학습 시작
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="stt_fastconformer")
def main(cfg: DictConfig):
    # 모델 초기화
    model = FastConformer(
        d_model=cfg.model.encoder.d_model,
        n_layers=cfg.model.encoder.n_layers,
        n_heads=cfg.model.encoder.n_heads,
        vocab_size=cfg.model.decoder.vocab_size
    )

    # 데이터로더 설정
    train_loader = get_dataloader(cfg.data.train_manifest, cfg)
    val_loader = get_dataloader(cfg.data.val_manifest, cfg, is_train=False)

    # Trainer 초기화 및 학습
    trainer = STTTrainer(model, train_loader, val_loader, cfg.training)
    trainer.train()

if __name__ == "__main__":
    main()
```

### MLOps

```python
# MLOps 스택
MLOPS_STACK = {
    "data_versioning": {
        "tool": "DVC",
        "why": "Git과 통합, 대용량 데이터 관리",
        "usage": """
        dvc add data/training_data.tar.gz
        dvc push  # S3로 업로드
        git add data/training_data.tar.gz.dvc
        git commit -m 'Add training data v2'
        """
    },
    "model_registry": {
        "tool": "MLflow Model Registry",
        "why": "모델 버전 관리, 스테이징/프로덕션 구분",
        "stages": ["None", "Staging", "Production", "Archived"]
    },
    "pipeline_orchestration": {
        "tool": "Airflow",
        "why": "스케줄링, 의존성 관리, 모니터링 UI",
        "example_dags": [
            "daily_data_preprocessing",
            "weekly_model_retraining",
            "hourly_evaluation_metrics"
        ]
    },
    "feature_store": {
        "tool": "Feast (선택적)",
        "why": "특성 재사용, 온라인/오프라인 일관성",
        "reality": "규모가 작으면 안 써도 됨"
    }
}
```

실제 Airflow DAG 예시입니다.

```python
# dags/model_training_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["ml-team@company.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "stt_model_retraining",
    default_args=default_args,
    description="Weekly STT model retraining pipeline",
    schedule_interval="0 2 * * 0",  # 매주 일요일 새벽 2시
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "stt", "training"],
) as dag:

    # 1. 새로운 데이터 수집
    collect_data = BashOperator(
        task_id="collect_new_data",
        bash_command="""
        python scripts/collect_data.py \
            --start-date {{ ds }} \
            --end-date {{ next_ds }} \
            --output /data/weekly/{{ ds }}/
        """
    )

    # 2. 데이터 전처리
    def preprocess_data(**context):
        from data_pipeline import preprocess
        ds = context["ds"]
        preprocess(
            input_dir=f"/data/weekly/{ds}/",
            output_dir=f"/data/processed/{ds}/"
        )

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    # 3. 모델 학습
    train_model = BashOperator(
        task_id="train_model",
        bash_command="""
        python train.py \
            --config configs/stt_production.yaml \
            --data-dir /data/processed/{{ ds }}/ \
            --experiment-name stt-weekly-{{ ds }}
        """,
        execution_timeout=timedelta(hours=24),
    )

    # 4. 모델 평가
    def evaluate_model(**context):
        from evaluation import evaluate
        ds = context["ds"]
        metrics = evaluate(
            model_path=f"/models/stt-weekly-{ds}/best.pt",
            test_manifest="/data/test_set.jsonl"
        )

        # XCom으로 메트릭 전달
        context["ti"].xcom_push(key="metrics", value=metrics)

        # 성능이 기준 이하면 실패 처리
        if metrics["cer"] > 0.05:  # CER 5% 초과시
            raise ValueError(f"Model CER too high: {metrics['cer']}")

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    # 5. 모델 배포 (조건부)
    deploy_model = BashOperator(
        task_id="deploy_model",
        bash_command="""
        python scripts/deploy.py \
            --model-path /models/stt-weekly-{{ ds }}/best.pt \
            --environment staging
        """
    )

    # 6. 슬랙 알림
    notify_success = SlackWebhookOperator(
        task_id="notify_success",
        slack_webhook_conn_id="slack_ml_alerts",
        message="""
        :white_check_mark: STT Model Retraining Complete
        - Date: {{ ds }}
        - CER: {{ ti.xcom_pull(task_ids='evaluate_model', key='metrics')['cer'] }}
        - Model deployed to staging
        """,
    )

    # Task 의존성
    collect_data >> preprocess >> train_model >> evaluate >> deploy_model >> notify_success
```

### 서빙 인프라

```python
# 서빙 인프라 스택
SERVING_STACK = {
    "inference_server": {
        "primary": "Triton Inference Server",
        "why": "다양한 프레임워크 지원, 동적 배치, GPU 활용 최적화",
        "alternatives": ["TorchServe", "TF Serving", "Ray Serve"]
    },
    "api_gateway": {
        "primary": "FastAPI",
        "why": "빠름, 타입 힌트, 자동 문서화",
        "alternatives": ["Flask", "gRPC 직접"]
    },
    "containerization": {
        "runtime": "Docker",
        "orchestration": "Kubernetes",
        "gpu_support": "NVIDIA Container Toolkit"
    },
    "optimization": {
        "quantization": "INT8 (TensorRT)",
        "format": "ONNX, TensorRT",
        "batching": "Dynamic Batching (Triton)"
    }
}
```

Triton 설정 예시입니다.

```protobuf
# model_repository/stt_model/config.pbtxt
name: "stt_model"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  {
    name: "audio"
    data_type: TYPE_FP32
    dims: [-1]  # variable length audio
  }
]

output [
  {
    name: "text"
    data_type: TYPE_STRING
    dims: [1]
  },
  {
    name: "confidence"
    data_type: TYPE_FP32
    dims: [1]
  }
]

# Dynamic batching 설정
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 100000  # 100ms
}

# 인스턴스 설정
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

# 모델 버전 관리
version_policy: { latest: { num_versions: 2 }}
```

### 모니터링

```python
# 모니터링 스택
MONITORING_STACK = {
    "metrics": {
        "collection": "Prometheus",
        "visualization": "Grafana",
        "custom_metrics": [
            "inference_latency_seconds",
            "inference_requests_total",
            "model_accuracy_score",
            "gpu_memory_usage_bytes"
        ]
    },
    "logging": {
        "aggregation": "ELK Stack / Loki",
        "structured_logging": "structlog",
    },
    "alerting": {
        "tool": "Grafana Alerting / PagerDuty",
        "alert_examples": [
            "latency_p99 > 500ms for 5 minutes",
            "error_rate > 1% for 10 minutes",
            "gpu_memory > 90% for 15 minutes"
        ]
    },
    "tracing": {
        "tool": "Jaeger / OpenTelemetry",
        "why": "요청 흐름 추적, 병목 지점 파악"
    }
}
```

실제 모니터링 코드 예시입니다.

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# 메트릭 정의
INFERENCE_REQUESTS = Counter(
    "stt_inference_requests_total",
    "Total number of inference requests",
    ["model_version", "status"]
)

INFERENCE_LATENCY = Histogram(
    "stt_inference_latency_seconds",
    "Inference latency in seconds",
    ["model_version"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

AUDIO_DURATION = Histogram(
    "stt_audio_duration_seconds",
    "Duration of audio being processed",
    buckets=[1, 5, 10, 30, 60, 120]
)

GPU_MEMORY = Gauge(
    "stt_gpu_memory_bytes",
    "GPU memory usage in bytes",
    ["gpu_id"]
)

MODEL_CER = Gauge(
    "stt_model_cer",
    "Current model Character Error Rate",
    ["model_version", "test_set"]
)

def track_inference(model_version: str):
    """추론 메트릭을 추적하는 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                INFERENCE_REQUESTS.labels(
                    model_version=model_version,
                    status="success"
                ).inc()
                return result
            except Exception as e:
                INFERENCE_REQUESTS.labels(
                    model_version=model_version,
                    status="error"
                ).inc()
                raise
            finally:
                latency = time.time() - start_time
                INFERENCE_LATENCY.labels(
                    model_version=model_version
                ).observe(latency)
        return wrapper
    return decorator

# 사용 예시
@track_inference(model_version="v1.2.0")
async def transcribe(audio: np.ndarray) -> dict:
    """음성을 텍스트로 변환"""
    AUDIO_DURATION.observe(len(audio) / 16000)  # 16kHz 가정
    result = await model.infer(audio)
    return result

# GPU 메모리 주기적 업데이트
def update_gpu_metrics():
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        GPU_MEMORY.labels(gpu_id=str(i)).set(mem_info.used)
```

## 필요한 스킬

정해진 건 없지만, 제 경험상 중요했던 것들을 정리해봅니다.

### 반드시 필요한 것

```python
# 기본 스킬셋
ESSENTIAL_SKILLS = {
    "pytorch_fundamentals": {
        "why": "모델 학습/평가 루프를 직접 짤 수 있어야 합니다",
        "specifics": [
            "Dataset, DataLoader 구현",
            "Custom Module 작성",
            "학습 루프 (forward, backward, optimizer step)",
            "체크포인트 저장/로드",
            "분산 학습 기본 (DDP)"
        ],
        "level": "코드 보고 이해 + 처음부터 작성 가능"
    },
    "experiment_management": {
        "why": "실험 관리 없이는 재현성이 없습니다",
        "tools": ["wandb", "mlflow", "tensorboard"],
        "specifics": [
            "하이퍼파라미터 로깅",
            "메트릭 시각화",
            "모델 비교",
            "실험 재현"
        ]
    },
    "linux_docker": {
        "why": "서버 환경에서 작업하니까요",
        "specifics": [
            "기본 Linux 명령어",
            "Dockerfile 작성",
            "docker-compose",
            "SSH, tmux/screen"
        ]
    }
}
```

기본기가 얼마나 중요한지 강조하고 싶습니다. 실제로 자주 쓰는 패턴들입니다.

```python
# 자주 사용하는 PyTorch 패턴들

# 1. Custom Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data = self._load_data(data_path)
        self.transform = transform

    def _load_data(self, path):
        # 실제로는 더 복잡한 로직
        import json
        with open(path) as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        return item

# 2. Custom Module
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape to (batch, n_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.out_proj(out)

        return out, attn_weights

# 3. 체크포인트 저장/로드
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
    }, path)

def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path, map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"]

# 4. DDP 학습 설정
def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    torch.distributed.destroy_process_group()

def train_ddp(rank, world_size, model, dataset):
    setup_ddp(rank, world_size)

    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    # 학습 루프...

    cleanup_ddp()
```

### 있으면 좋은 것

```python
# 중급 스킬셋
NICE_TO_HAVE = {
    "serving": {
        "why": "모델을 서비스에 넣으려면 필요합니다",
        "tools": ["FastAPI", "Triton", "TorchServe", "ONNX"],
        "specifics": [
            "REST API 설계",
            "모델 최적화 (양자화, 프루닝)",
            "배치 처리",
            "지연시간 프로파일링"
        ]
    },
    "distributed_training": {
        "why": "큰 모델을 학습하려면 필요합니다",
        "tools": ["DeepSpeed", "FSDP", "Ray"],
        "specifics": [
            "Data Parallelism (DDP)",
            "Model Parallelism",
            "ZeRO optimization",
            "Mixed Precision Training"
        ]
    },
    "paper_implementation": {
        "why": "최신 기술을 빠르게 적용하려면",
        "specifics": [
            "논문의 핵심 아이디어 파악",
            "공식 코드 없이 구현",
            "기존 코드베이스에 통합"
        ]
    }
}
```

모델 최적화 예시입니다.

```python
# 모델 최적화 예시

# 1. ONNX 변환
def export_to_onnx(model, sample_input, output_path):
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        input_names=["audio"],
        output_names=["logits"],
        dynamic_axes={
            "audio": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=14
    )
    print(f"ONNX model exported to {output_path}")

# 2. TensorRT 최적화
def optimize_with_tensorrt(onnx_path, trt_path, precision="fp16"):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # ONNX 파싱
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    # 빌드 설정
    config = builder.create_builder_config()
    config.max_workspace_size = 4 << 30  # 4GB

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # INT8 calibration 필요

    # Dynamic shape 설정
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "audio",
        min=(1, 1000),
        opt=(16, 160000),
        max=(32, 320000)
    )
    config.add_optimization_profile(profile)

    # 엔진 빌드
    engine = builder.build_engine(network, config)

    with open(trt_path, "wb") as f:
        f.write(engine.serialize())

    print(f"TensorRT engine saved to {trt_path}")

# 3. 양자화 (PyTorch Native)
def quantize_model(model, calibration_data):
    model.eval()

    # 동적 양자화 (간단)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM},
        dtype=torch.qint8
    )

    return quantized_model

# 4. 정적 양자화 (더 정확함)
def static_quantize(model, calibration_loader):
    model.eval()

    # Fuse 가능한 레이어 합치기
    model = torch.quantization.fuse_modules(model, [
        ['conv', 'bn', 'relu'],
    ])

    # QConfig 설정
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Prepare
    torch.quantization.prepare(model, inplace=True)

    # Calibration
    with torch.no_grad():
        for batch in calibration_loader:
            model(batch)

    # Convert
    torch.quantization.convert(model, inplace=True)

    return model
```

### 의외로 중요한 것

```python
# 소프트 스킬
SURPRISINGLY_IMPORTANT = {
    "communication": {
        "why": "비개발자에게 기술을 설명해야 합니다",
        "examples": [
            "CEO에게 왜 이 모델이 더 좋은지 설명",
            "PM에게 배포 일정이 왜 늦어지는지 설명",
            "마케팅팀에게 AI 기능의 한계 설명"
        ],
        "tip": "전문 용어 없이 비유로 설명하는 연습"
    },
    "prioritization": {
        "why": "할 일이 항상 넘쳐납니다",
        "framework": {
            "urgent_important": "지금 당장",
            "not_urgent_important": "일정 잡아서",
            "urgent_not_important": "다른 사람에게 위임",
            "not_urgent_not_important": "안 함"
        },
        "reality": "스타트업에서는 모든 게 긴급해 보여서 판단이 어려움"
    },
    "learning_speed": {
        "why": "기술이 너무 빨리 바뀝니다",
        "examples": [
            "2023년: LLM이 대세",
            "2024년: RAG, Agent가 핫함",
            "2025년: ???"
        ],
        "approach": "트렌드를 따르되, 기본기를 소홀히 하지 않기"
    }
}
```

비개발자에게 기술 설명하는 실제 예시입니다.

```python
# 비유를 사용한 기술 설명 예시

EXPLANATIONS_FOR_NON_TECH = {
    "why_model_needs_more_data": {
        "bad": "데이터가 부족해서 과적합되고 있습니다",
        "good": """
        아이에게 사과 사진 10장만 보여주면 빨간 것만 사과로 알아요.
        하지만 1000장을 보여주면 초록 사과도 알게 되죠.
        지금 우리 모델이 그런 상태입니다.
        """
    },
    "why_model_is_slow": {
        "bad": "모델 파라미터가 많아서 추론 latency가 높습니다",
        "good": """
        두꺼운 책을 읽는 것과 얇은 책을 읽는 것의 차이예요.
        두꺼운 책이 더 정확하지만 시간이 오래 걸리죠.
        지금 최적화 작업은 중요한 내용만 남기고
        책을 얇게 만드는 거예요.
        """
    },
    "why_same_input_different_output": {
        "bad": "temperature 파라미터 때문에 샘플링이 다르게 됩니다",
        "good": """
        같은 질문에도 사람마다 다르게 대답하잖아요?
        AI도 마찬가지로 창의성 설정이 있어요.
        매번 같은 답이 나오게 하려면 창의성을 0으로 하면 됩니다.
        """
    },
    "why_model_confidently_wrong": {
        "bad": "LLM의 hallucination 문제입니다",
        "good": """
        시험 볼 때 모르는 문제도 일단 답을 쓰는 학생 있잖아요?
        AI도 비슷해요. 모른다고 말하는 법을 안 배웠거든요.
        그래서 확실해 보이지만 틀린 답을 하는 거예요.
        """
    }
}
```

## 하루 일과 예시

실제 하루가 어떻게 흘러가는지 공유합니다.

```python
# 일반적인 하루 (바쁘지 않을 때)
TYPICAL_DAY_NORMAL = {
    "09:00-10:00": {
        "activity": "출근, 메일/슬랙 확인",
        "details": "밤새 돌린 실험 결과 확인, 이슈 트래커 확인"
    },
    "10:00-12:00": {
        "activity": "집중 업무 (오전)",
        "details": "코딩, 실험, 논문 읽기 등 딥워크"
    },
    "12:00-13:00": {
        "activity": "점심",
        "details": "팀원들과 식사하며 가벼운 대화"
    },
    "13:00-14:00": {
        "activity": "미팅 또는 가벼운 업무",
        "details": "스탠드업, 1:1, 코드 리뷰"
    },
    "14:00-18:00": {
        "activity": "집중 업무 (오후)",
        "details": "오전에 하던 작업 계속"
    },
    "18:00-19:00": {
        "activity": "정리",
        "details": "실험 돌려놓기, 내일 할 일 정리, 퇴근"
    }
}

# 바쁠 때 (출시 전)
TYPICAL_DAY_CRUNCH = {
    "09:00-23:00": {
        "activity": "버그 수정, 최적화, 테스트",
        "details": "식사도 대충, 집중해서 작업",
        "frequency": "출시 전 1-2주"
    }
}
```

### 일주일 단위 업무 흐름

하루 단위보다 일주일 단위로 보는 게 더 현실적입니다.

```python
# 전형적인 한 주
TYPICAL_WEEK = {
    "monday": {
        "focus": "주간 계획, 코드 리뷰",
        "meetings": ["Weekly standup", "Sprint planning"],
        "tasks": [
            "지난주 실험 결과 정리",
            "이번 주 목표 설정",
            "PR 리뷰"
        ]
    },
    "tuesday": {
        "focus": "딥 워크 (구현)",
        "meetings": ["Minimal"],
        "tasks": [
            "새로운 기능 구현",
            "모델 학습 시작"
        ]
    },
    "wednesday": {
        "focus": "딥 워크 (구현)",
        "meetings": ["Tech sync (30분)"],
        "tasks": [
            "화요일 작업 계속",
            "학습 중인 모델 중간 점검"
        ]
    },
    "thursday": {
        "focus": "리뷰, 디버깅",
        "meetings": ["1:1 (매니저)", "Design review (필요시)"],
        "tasks": [
            "코드 리뷰 처리",
            "버그 수정",
            "문서화"
        ]
    },
    "friday": {
        "focus": "마무리, 학습",
        "meetings": ["Demo/Show & Tell (격주)"],
        "tasks": [
            "이번 주 작업 마무리",
            "논문 읽기 / 새로운 기술 학습",
            "주말 동안 돌릴 실험 설정"
        ]
    }
}
```

## 취업/이직 준비

AI 분야 취업이나 이직을 준비하는 분들께 도움이 될 만한 내용입니다.

### 포트폴리오

```python
# 좋은 포트폴리오의 조건
GOOD_PORTFOLIO = {
    "has_numbers": {
        "bad": "모델 성능을 개선했다",
        "good": "latency를 200ms에서 50ms로 75% 줄였다",
        "better": "latency를 200ms→50ms로 줄여서, 동시 처리량이 4배 증가"
    },
    "shows_depth": {
        "bad": "Hugging Face 모델을 파인튜닝했다",
        "good": "LoRA로 파인튜닝해서 메모리 사용량 70% 감소",
        "better": "LoRA + 양자화 조합으로 RTX 3060에서도 추론 가능하게"
    },
    "explains_why": {
        "bad": "Transformer를 사용했다",
        "good": "RNN 대비 병렬화가 가능해서 학습 속도 3배",
        "better": "RNN → Transformer로 바꾸고 학습 시간 72시간→24시간, 정확도도 2% 향상"
    }
}
```

실제 포트폴리오 작성 예시입니다.

```markdown
## 프로젝트: 한국어 음성인식 엔진 개발

### 개요
FastConformer 기반 한국어 STT 엔진 개발. 기존 상용 API 대비 비용 90% 절감,
정확도 동등 수준 달성.

### 역할
- 팀 규모: 3명 중 ML 리드
- 기여도: 모델 아키텍처 설계 70%, 학습 파이프라인 100%, 서빙 50%

### 기술 스택
- 모델: PyTorch, NeMo, FastConformer-CTC
- 데이터: KsponSpeech (1000h) + 자체 데이터 (500h)
- 서빙: Triton, TensorRT, FastAPI
- 인프라: K8s, Prometheus, Grafana

### 주요 성과

| 지표 | Before | After | 개선율 |
|------|--------|-------|--------|
| CER (한국어) | 8.2% | 3.1% | 62% ↓ |
| Latency (P99) | 350ms | 120ms | 66% ↓ |
| 비용/시간 | $0.024 | $0.002 | 92% ↓ |
| 동시 처리량 | 10 req/s | 100 req/s | 10x |

### 기술적 도전과 해결

**문제 1: 한국어 받침 인식 오류 높음**
- 원인: 기존 토크나이저가 한국어 자모 구조를 무시
- 해결: 자모 분리 기반 커스텀 토크나이저 구현
- 결과: 받침 오류율 15% → 3%

**문제 2: 긴 오디오에서 메모리 부족**
- 원인: Self-attention의 O(n²) 메모리 복잡도
- 해결: Chunked attention 구현 + 스트리밍 추론
- 결과: 1시간 오디오도 8GB GPU에서 처리 가능

**문제 3: 프로덕션 latency 목표 미달**
- 원인: FP32 추론, Python GIL
- 해결: TensorRT FP16 변환, Triton 배치 처리
- 결과: 350ms → 120ms

### 코드 샘플
GitHub: github.com/username/korean-stt (오픈소스화 불가시 private)
```

### 면접 준비

면접에서 자주 받는 질문들입니다.

```python
# 자주 나오는 질문
COMMON_INTERVIEW_QUESTIONS = {
    "project_deep_dive": [
        "이 프로젝트에서 가장 어려웠던 점은?",
        "왜 그 방법을 선택했나?",
        "다시 한다면 뭘 다르게 하겠나?",
        "실패했던 시도는 뭐가 있었나?"
    ],
    "technical": [
        "Transformer의 attention은 어떻게 동작하나?",
        "Batch Normalization과 Layer Normalization의 차이는?",
        "학습이 안 될 때 어떻게 디버깅하나?",
        "과적합을 어떻게 방지하나?"
    ],
    "system_design": [
        "STT 서비스를 설계한다면?",
        "추천 시스템의 아키텍처를 설명해보세요",
        "실시간 추론 시스템에서 고려할 점은?"
    ],
    "behavioral": [
        "팀원과 의견 충돌이 있으면 어떻게 해결하나?",
        "마감이 촉박할 때 우선순위를 어떻게 정하나?",
        "새로운 기술을 어떻게 학습하나?"
    ]
}
```

기술 면접에서 자주 묻는 개념들의 간단한 답변 예시입니다.

```python
# 기술 면접 답변 예시

TECHNICAL_ANSWERS = {
    "transformer_attention": """
    Q: Transformer의 self-attention은 어떻게 동작하나요?

    A: Self-attention은 시퀀스 내 모든 위치 간의 관계를 계산합니다.

    1. 입력 임베딩에서 Q(Query), K(Key), V(Value) 행렬을 만듭니다
    2. Q와 K의 내적으로 attention score를 계산합니다
    3. Score를 √d_k로 나눠 스케일링합니다 (gradient 안정화)
    4. Softmax로 확률 분포를 만듭니다
    5. 이 확률로 V를 가중합하여 출력을 만듭니다

    장점: 병렬화 가능, 긴 거리 의존성 학습
    단점: O(n²) 메모리/연산 복잡도
    """,

    "bn_vs_ln": """
    Q: Batch Norm과 Layer Norm의 차이는?

    A: 정규화하는 축이 다릅니다.

    Batch Norm: 배치 차원으로 정규화
    - (N, C, H, W)에서 N 축으로 평균/분산 계산
    - 배치 크기에 의존 → 작은 배치에서 불안정
    - 주로 CNN에서 사용

    Layer Norm: 특성 차원으로 정규화
    - 각 샘플 내에서 평균/분산 계산
    - 배치 크기와 무관 → 시퀀스 길이 가변에 적합
    - 주로 Transformer/RNN에서 사용

    실무에서: Transformer 계열은 LayerNorm이 표준입니다.
    """,

    "debugging_training": """
    Q: 모델 학습이 안 될 때 어떻게 디버깅하나요?

    A: 단계별로 체크합니다:

    1. 데이터 확인
       - 샘플 몇 개 직접 눈으로 확인
       - 라벨이 제대로 달렸는지
       - 전처리가 올바른지

    2. 모델 확인
       - 작은 데이터셋에서 과적합되는지 (안 되면 모델 버그)
       - Gradient가 흐르는지 (vanishing/exploding)
       - 초기 loss가 예상 범위인지

    3. 학습 설정 확인
       - Learning rate가 적절한지
       - Batch size 조정
       - Optimizer/Scheduler 점검

    실제 경험: 90%는 데이터 문제였습니다.
    """
}
```

```python
# 좋은 답변의 구조 (STAR 방법)
STAR_METHOD = {
    "Situation": "상황 설명",
    "Task": "내가 맡은 역할/목표",
    "Action": "내가 한 구체적 행동",
    "Result": "결과 (숫자로)"
}

# 예시
ANSWER_EXAMPLE = {
    "question": "가장 어려웠던 프로젝트는?",
    "bad_answer": "STT 개발이 어려웠어요. 성능이 안 나왔는데 열심히 해서 개선했어요.",
    "good_answer": """
    [Situation] FastConformer 기반 한국어 STT를 개발했는데,
    초기 CER이 8%로 목표치 4%에 한참 못 미쳤습니다.

    [Task] 2주 안에 CER 4% 이하로 낮추는 게 제 목표였습니다.

    [Action]
    1. 데이터 분석 결과 받침 인식 오류가 가장 컸습니다
    2. 자모 분리 로직을 직접 구현해서 전처리 개선
    3. 학습 데이터 중 노이즈 많은 것 필터링

    [Result] CER 8% → 3.1%로 개선, 목표 초과 달성.
    이 경험으로 '데이터 전처리가 모델 아키텍처만큼 중요하다'는 걸 배웠습니다.
    """
}
```

### 시스템 디자인 면접

ML 시스템 디자인 질문도 자주 나옵니다.

```python
# 시스템 디자인 답변 프레임워크
SYSTEM_DESIGN_FRAMEWORK = {
    "1_clarify_requirements": [
        "예상 QPS는? (초당 요청 수)",
        "지연시간 요구사항은? (P99 기준)",
        "정확도 요구사항은?",
        "예산 제약은?",
        "실시간 vs 배치?"
    ],
    "2_high_level_design": [
        "전체 아키텍처 다이어그램",
        "주요 컴포넌트 식별",
        "데이터 흐름 설명"
    ],
    "3_deep_dive": [
        "모델 선택 및 근거",
        "학습 파이프라인",
        "서빙 아키텍처",
        "확장성 고려사항"
    ],
    "4_trade_offs": [
        "정확도 vs 지연시간",
        "비용 vs 성능",
        "복잡도 vs 유지보수성"
    ]
}

# 예시: STT 서비스 설계
STT_SYSTEM_DESIGN = """
Q: 하루 100만 건의 음성 파일을 처리하는 STT 서비스를 설계해보세요.

[요구사항 확인]
- QPS: 100만/일 ≈ 12 req/s (평균), 피크시 10배 = 120 req/s
- Latency: P99 < 2초 (10초 오디오 기준)
- 정확도: CER < 5%
- 비용: 건당 $0.01 이하

[High-Level Design]

                    ┌─────────────┐
                    │   Client    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ API Gateway │
                    │  (FastAPI)  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌────▼────┐ ┌────▼────┐
        │ Worker 1  │ │Worker 2 │ │Worker N │
        │(Triton)   │ │(Triton) │ │(Triton) │
        └─────┬─────┘ └────┬────┘ └────┬────┘
              │            │            │
              └────────────┴────────────┘
                           │
                    ┌──────▼──────┐
                    │   Storage   │
                    │    (S3)     │
                    └─────────────┘

[컴포넌트 상세]

1. API Gateway (FastAPI)
   - 요청 유효성 검사
   - Rate limiting
   - 로드 밸런싱

2. Worker (Triton + GPU)
   - FastConformer 모델
   - Dynamic batching
   - Auto-scaling (HPA)

3. 비동기 처리 (긴 오디오용)
   - Kafka/SQS로 큐잉
   - 완료시 Webhook 콜백

[확장성]
- 수평 확장: K8s HPA로 worker 수 자동 조절
- GPU: 피크시 10개, 평시 3개
- 비용 최적화: Spot instance 활용

[모니터링]
- 메트릭: Latency, Error rate, GPU utilization
- 알림: P99 > 3초, Error rate > 1%

[비용 예측]
- GPU (A10G): $1.2/시간 × 5대 평균 = $144/일
- 요청당: $144 / 100만 = $0.000144 ✓ (목표 달성)
"""
```

### 실패 경험 이야기하기

실패 경험을 이야기하는 것을 두려워하지 마세요. 오히려 좋은 인상을 줄 수 있습니다.

```python
# 실패 경험의 가치
FAILURE_EXPERIENCE = {
    "why_valuable": [
        "문제 해결 능력을 보여줌",
        "겸손함과 성장 마인드셋",
        "실제로 도전해본 경험"
    ],
    "how_to_tell": {
        "structure": [
            "무엇을 시도했는지",
            "왜 실패했는지 (원인 분석)",
            "그래서 뭘 배웠는지",
            "다음에는 어떻게 할 건지"
        ],
        "example": """
        처음에 Whisper를 그대로 파인튜닝했는데 한국어 성능이 안 나왔습니다.
        원인을 분석해보니, Whisper의 토크나이저가 한국어에 최적화되어 있지 않았습니다.
        이후 FastConformer로 바꾸고, 한국어 자모 기반 토크나이저를 직접 만들어서
        CER을 8%에서 3.1%로 개선했습니다.
        이 경험으로 '모델 아키텍처보다 데이터와 토크나이징이 더 중요할 수 있다'는 걸 배웠습니다.
        """
    },
    "red_flags": [
        "실패를 남 탓으로 돌림",
        "실패에서 배운 게 없음",
        "실패 경험이 아예 없다고 함 (비현실적)"
    ]
}
```

## 연봉과 보상

솔직하게 이야기해보겠습니다.

```python
# 2024년 기준 국내 AI/ML 엔지니어 연봉 (추정)
SALARY_RANGES_2024 = {
    "junior": {  # 0-3년차
        "big_company": {
            "base": "5,000만원 - 7,000만원",
            "bonus": "연봉의 0-20%",
            "stock": "RSU (대기업의 경우)"
        },
        "startup": {
            "base": "5,000만원 - 6,500만원",
            "bonus": "0-10%",
            "stock": "스톡옵션 (가치 불확실)"
        }
    },
    "mid": {  # 3-6년차
        "big_company": {
            "base": "7,000만원 - 1억원",
            "bonus": "연봉의 10-30%",
            "stock": "RSU"
        },
        "startup": {
            "base": "6,500만원 - 9,000만원",
            "bonus": "0-15%",
            "stock": "스톡옵션 (비중 높음)"
        }
    },
    "senior": {  # 6년+
        "big_company": {
            "base": "1억원 - 1.5억원",
            "bonus": "연봉의 15-40%",
            "stock": "RSU (상당량)"
        },
        "startup": {
            "base": "9,000만원 - 1.3억원",
            "bonus": "0-20%",
            "stock": "스톡옵션 (대량)"
        }
    },
    "disclaimer": """
    * 회사마다, 개인마다 편차가 큽니다
    * 특히 스타트업은 협상에 따라 크게 달라집니다
    * 외국계는 보통 30-50% 더 높습니다
    """
}

# 스톡옵션에 대한 현실적인 시각
STOCK_OPTIONS_REALITY = {
    "best_case": {
        "scenario": "회사가 IPO 또는 대기업에 인수",
        "outcome": "수억원 ~ 수십억원 가능",
        "probability": "매우 낮음 (< 5%)"
    },
    "average_case": {
        "scenario": "회사가 계속 성장하지만 exit 없음",
        "outcome": "종이 위의 가치만 존재",
        "probability": "대부분"
    },
    "worst_case": {
        "scenario": "회사 망함",
        "outcome": "휴지조각",
        "probability": "스타트업의 90%가 실패"
    },
    "my_advice": """
    스톡옵션은 '없다고 생각하고' 연봉을 협상하세요.
    그래야 나중에 실망하지 않습니다.
    만약 성공하면 그건 보너스입니다.
    """
}
```

## 대기업 vs 스타트업 비교

솔직한 비교입니다.

```python
# 객관적 비교
COMPARISON = {
    "stability": {
        "big_company": {
            "pros": "안정적 급여, 복지, 고용 안정성",
            "cons": "성장 정체 가능성"
        },
        "startup": {
            "pros": "성공 시 큰 보상 (스톡옵션)",
            "cons": "망할 수 있음, 급여 불안정"
        }
    },
    "learning": {
        "big_company": {
            "pros": "체계적 온보딩, 시니어에게 배움, 전문화",
            "cons": "넓은 경험 어려움, 내 역할만"
        },
        "startup": {
            "pros": "다양한 경험, 빠른 성장, 오너십",
            "cons": "체계 부족, 스스로 해결해야"
        }
    },
    "impact": {
        "big_company": {
            "pros": "대규모 서비스 경험",
            "cons": "내 기여가 잘 안 보임"
        },
        "startup": {
            "pros": "내가 만든 게 바로 서비스됨",
            "cons": "서비스 규모가 작을 수 있음"
        }
    },
    "work_life_balance": {
        "big_company": {
            "pros": "정시 퇴근 가능, 예측 가능한 일정",
            "cons": "회사마다 다름"
        },
        "startup": {
            "pros": "유연한 근무 (재량)",
            "cons": "야근이 일상인 시기도 있음"
        }
    }
}
```

## 스타트업의 현실적인 단점

스타트업이 다 좋은 건 아닙니다. 솔직하게 얘기하자면:

```python
# 스타트업의 현실
STARTUP_REALITY = {
    "instability": {
        "issue": "언제 망할지 모릅니다",
        "my_experience": "주변에 망한 스타트업 5곳 이상",
        "mitigation": [
            "재무 상태 확인 (런웨이)",
            "투자 라운드 확인",
            "비즈니스 모델 검토"
        ]
    },
    "lack_of_structure": {
        "issue": "온보딩? 문서화? 그런 거 없습니다",
        "my_experience": "첫날부터 '이거 해줘' 받음",
        "reality": [
            "알아서 코드 파악해야 함",
            "물어볼 사람이 바쁨",
            "문서가 없거나 outdated"
        ]
    },
    "workload": {
        "issue": "바쁠 땐 진짜 바쁩니다",
        "my_experience": "출시 전 2주는 매일 야근",
        "honest_truth": "워라밸은 시기에 따라 크게 변동"
    },
    "pressure": {
        "issue": "내가 안 하면 아무도 안 합니다",
        "feeling": "대기업에서 몰랐던 종류의 스트레스",
        "examples": [
            "서비스 장애 시 내가 해결해야",
            "성능 문제도 내가 해결해야",
            "신기술 도입도 내가 조사해야"
        ]
    }
}
```

### 스타트업 선택 시 체크리스트

```python
# 스타트업 평가 체크리스트
STARTUP_EVALUATION_CHECKLIST = {
    "business": {
        "questions": [
            "런웨이가 얼마나 남았나? (최소 12개월 이상)",
            "수익이 나고 있나? / 수익 모델이 명확한가?",
            "시장 규모는? 성장 가능성은?",
            "경쟁사 대비 차별점은?"
        ],
        "red_flags": [
            "런웨이 6개월 미만",
            "수익 모델 불명확",
            "'그냥 좋은 제품 만들면 된다'"
        ]
    },
    "team": {
        "questions": [
            "창업자/경영진 배경은?",
            "ML 팀 규모와 구성은?",
            "시니어가 있나?",
            "팀 이직률은?"
        ],
        "red_flags": [
            "창업자 첫 창업 + 경험 부족",
            "높은 이직률",
            "ML 팀이 1-2명뿐"
        ]
    },
    "technology": {
        "questions": [
            "기술 스택은 현대적인가?",
            "기술 부채 수준은?",
            "R&D 투자 비중은?",
            "특허/IP가 있나?"
        ],
        "red_flags": [
            "레거시 기술 스택",
            "문서화 전무",
            "테스트 없음"
        ]
    },
    "culture": {
        "questions": [
            "의사결정 방식은?",
            "실패를 어떻게 다루나?",
            "성장 기회가 있나?",
            "워라밸은 어떤가?"
        ],
        "red_flags": [
            "마이크로매니징",
            "실패 = 비난",
            "'열정적인' 강조 (야근의 다른 표현)"
        ]
    },
    "compensation": {
        "questions": [
            "기본급은 시장 수준인가?",
            "스톡옵션 조건은? (행사가, 베스팅)",
            "보너스 구조는?",
            "복지는?"
        ],
        "red_flags": [
            "기본급 심하게 낮고 '스톡옵션으로 보상'",
            "스톡옵션 조건 불명확",
            "복지 거의 없음"
        ]
    }
}
```

## 커리어 성장 경로

ML 엔지니어의 커리어 경로에 대해 정리해봤습니다.

```python
# 커리어 경로 옵션
CAREER_PATHS = {
    "individual_contributor": {
        "path": "Junior → Mid → Senior → Staff → Principal",
        "focus": "기술적 깊이, 아키텍처, 멘토링",
        "pros": "기술에 집중 가능, 코딩 계속",
        "cons": "임팩트 범위 제한될 수 있음",
        "fit_for": "기술 자체를 좋아하는 사람"
    },
    "management": {
        "path": "Senior → Tech Lead → Manager → Director",
        "focus": "팀 빌딩, 프로젝트 관리, 전략",
        "pros": "더 큰 임팩트, 조직 성장",
        "cons": "코딩 시간 감소, 회의 많아짐",
        "fit_for": "사람과 일하는 걸 좋아하는 사람"
    },
    "entrepreneurship": {
        "path": "경험 쌓기 → 창업 또는 초기 멤버 합류",
        "focus": "제품, 비즈니스, 모든 것",
        "pros": "가장 큰 오너십, 성공 시 큰 보상",
        "cons": "리스크 높음, 스트레스",
        "fit_for": "뭔가 만들고 싶은 사람"
    },
    "research": {
        "path": "Industry → PhD → Research Scientist",
        "focus": "새로운 알고리즘, 논문",
        "pros": "최신 기술 탐구, 학문적 성취",
        "cons": "실용적 임팩트 늦음, 연봉 제한",
        "fit_for": "순수 연구를 좋아하는 사람"
    },
    "consulting": {
        "path": "경험 쌓기 → 프리랜서/컨설턴트",
        "focus": "여러 회사 프로젝트",
        "pros": "자유로움, 높은 단가 가능",
        "cons": "불안정, 영업 필요",
        "fit_for": "다양한 경험을 원하는 사람"
    }
}
```

## 결론

연구 엔지니어로 스타트업에 가려면, "연구만 하고 싶다"는 마인드로는 힘듭니다. 연구 + 엔지니어링 + 서비스 감각까지 필요합니다.

```python
# 어떤 사람에게 맞는가
FIT_ASSESSMENT = {
    "startup_fits_you_if": [
        "빠른 변화와 불확실성을 즐기는 편",
        "다양한 역할을 해보고 싶음",
        "내가 만든 게 서비스되는 걸 보고 싶음",
        "성장 속도를 중시함",
        "어느 정도 리스크를 감수할 수 있음"
    ],
    "big_company_fits_you_if": [
        "안정성과 예측 가능성을 중시함",
        "한 분야를 깊이 파고 싶음",
        "체계적인 환경에서 일하고 싶음",
        "워라밸이 매우 중요함",
        "대규모 시스템을 경험하고 싶음"
    ],
    "no_right_answer": """
    정답은 없습니다.
    뭘 원하느냐에 따라 다릅니다.

    대기업에서 기본기 쌓고 → 스타트업으로 가는 경로도 좋고,
    스타트업에서 빠르게 배우고 → 대기업으로 가는 경로도 좋습니다.
    """
}
```

대신 성장 속도는 빠릅니다. 대기업에서 3년 걸릴 경험을 1년에 할 수 있습니다. 물론 그만큼 힘들지만요.

### 마지막으로

제가 이 글을 쓰는 이유는, 저도 이직을 준비할 때 이런 정보가 없어서 막막했기 때문입니다. 누군가에게는 스타트업이 맞고, 누군가에게는 대기업이 맞습니다. 중요한 건 자신이 뭘 원하는지 아는 것입니다.

```python
# 자기 점검 질문
SELF_REFLECTION = {
    "values": [
        "5년 후에 어떤 사람이 되고 싶은가?",
        "일에서 가장 중요하게 생각하는 것은?",
        "얼마나 리스크를 감수할 수 있는가?",
        "워라밸이 얼마나 중요한가?"
    ],
    "practical": [
        "재정적 여유가 있는가? (스타트업 실패 대비)",
        "가족 상황은? (부양가족이 있다면 안정성 중요)",
        "현재 경력 단계는? (주니어라면 대기업에서 기초 쌓기도 좋음)"
    ],
    "my_answer": """
    저는 '내가 만든 게 쓰이는 걸 보고 싶다'가 가장 컸습니다.
    그래서 스타트업을 선택했고, 지금까지 후회 없습니다.
    하지만 이건 저의 선택일 뿐입니다.

    여러분의 선택은 여러분만이 할 수 있습니다.
    """
}
```

질문이 있으시면 편하게 연락 주세요.

## 참고 자료

### 기술 학습
- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)
- [Hugging Face Course](https://huggingface.co/course)
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
- [MLOps Community](https://mlops.community/)

### 커리어
- [Hacker News - Ask HN: What's it like to work at a startup?](https://news.ycombinator.com/item?id=37981002)
- [Glassdoor - AI/ML Engineer Salaries](https://www.glassdoor.com/)
- [Levels.fyi - Tech Compensation](https://www.levels.fyi/)
- [The Missing README: A Guide for the New Software Engineer](https://www.amazon.com/Missing-README-Guide-Software-Engineer/dp/1718501838)
- [Staff Engineer: Leadership Beyond the Management Track](https://staffeng.com/book)

### 면접 준비
- [Machine Learning Interviews Book](https://huyenchip.com/ml-interviews-book/)
- [Designing Machine Learning Systems (Chip Huyen)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
- [System Design Interview for ML Engineers](https://www.educative.io/courses/machine-learning-system-design)

