---
title: "FastConformer로 STT 엔진 만들면서 배운 것들"
description: "한국어/일본어 음성인식 모델 개발 과정에서 겪은 문제와 해결 방법"
date: 2024-10-01
slug: /fastconformer-stt-development
tags: [ai, dev]
published: true
---

# FastConformer로 STT 엔진 만들면서 배운 것들

Return Zero에서 STT 엔진 개발을 맡게 되었습니다. 한국어와 일본어를 동시에 지원해야 했는데, 기존 모델로는 만족스러운 성능이 나오지 않아서 FastConformer를 도입하게 되었습니다. 그 과정에서 겪은 문제들과 해결 방법을 공유합니다.

## 왜 FastConformer였나

처음에는 Whisper를 고려했습니다. OpenAI에서 공개한 모델이고, 성능도 좋다고 알려져 있으니까요. 하지만 Whisper에는 치명적인 문제가 있었습니다.

```python
# Whisper의 문제: 전체 오디오를 받아야 처리 시작
import whisper

model = whisper.load_model("base")

# 30초 오디오 전체를 받은 후에야 처리 가능
result = model.transcribe("audio.wav")

# 실시간 서비스에서 이러면 30초 지연 발생
# 사용자가 말 끝나고 30초 기다려야 텍스트가 나온다
```

저희 서비스는 실시간으로 자막을 보여줘야 했습니다. Whisper처럼 전체 오디오를 다 받은 다음에 처리하면 사용자 경험이 매우 나빠집니다.

```python
# 스트리밍 STT 요구사항
REQUIREMENTS = {
    "latency": {
        "target": "<500ms",  # 말한 지 0.5초 내에 텍스트
        "whisper": "30000ms",  # Whisper는 전체 오디오 길이만큼 대기
        "verdict": "불가능"
    },
    "realtime_factor": {
        "target": "<0.3",  # 1초 오디오를 0.3초 안에 처리
        "description": "RTF = 처리시간 / 오디오시간"
    },
    "memory": {
        "target": "<8GB",  # 서버 1대에 여러 요청 동시 처리
        "description": "동시 요청 수 = GPU 메모리 / 모델 메모리"
    }
}
```

[FastConformer](https://arxiv.org/abs/2305.05084)는 NVIDIA에서 발표한 모델로, 스트리밍 처리가 가능합니다. 오디오가 들어오는 대로 바로바로 텍스트로 변환할 수 있습니다.

```python
# FastConformer 스트리밍 처리 개념
class StreamingFastConformer:
    """
    FastConformer의 핵심 장점:
    1. 청크 단위 처리 - 전체 오디오 기다리지 않음
    2. 캐시 활용 - 이전 컨텍스트 재계산 없이 재사용
    3. 낮은 지연시간 - 실시간 서비스에 적합
    """

    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
        self.encoder_cache = None
        self.decoder_cache = None

    def process_chunk(self, audio_chunk: np.ndarray) -> str:
        """
        청크 단위로 오디오를 처리합니다.

        Args:
            audio_chunk: 1.6초 분량의 오디오 (16kHz 기준 25600 샘플)

        Returns:
            해당 구간의 텍스트
        """
        # 인코더: 오디오 → 특징 벡터
        # 캐시를 활용해 이전 청크 정보 유지
        features, self.encoder_cache = self.model.encode(
            audio_chunk,
            cache=self.encoder_cache
        )

        # 디코더: 특징 벡터 → 텍스트
        text, self.decoder_cache = self.model.decode(
            features,
            cache=self.decoder_cache
        )

        return text
```

## NeMo 프레임워크 활용

FastConformer를 직접 구현하는 것보다 NVIDIA의 [NeMo](https://github.com/NVIDIA/NeMo) 프레임워크를 활용하는 것이 효율적이었습니다.

```python
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

# 사전학습된 FastConformer 모델 로드
model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
    "nvidia/parakeet-ctc-1.1b"  # 1.1B 파라미터
)

# 또는 더 작은 모델
model_small = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
    "nvidia/parakeet-ctc-0.6b"  # 600M 파라미터
)

# 모델 크기별 비교
MODEL_COMPARISON = {
    "parakeet-ctc-0.6b": {
        "params": "600M",
        "memory": "2.4GB",
        "WER_librispeech": "2.5%",
        "use_case": "실시간 서비스, 리소스 제한 환경"
    },
    "parakeet-ctc-1.1b": {
        "params": "1.1B",
        "memory": "4.4GB",
        "WER_librispeech": "1.8%",
        "use_case": "정확도 중시, 배치 처리"
    }
}
```

## 삽질 기록

### 1. 한국어 자모 분리 문제

한국어 STT에서 가장 먼저 부딪힌 문제는 자모 분리였습니다. "감사합니다"를 "간사합니다"로 인식하는 등의 오류가 빈번했습니다.

```python
# 문제 상황: 자모 분리 방식에 따른 성능 차이
JAMO_ISSUES = {
    "original_text": "감사합니다",
    "wrong_recognition": "간사합니다",
    "cause": "받침 'ㅁ'과 'ㄴ'의 음향적 유사성 + 잘못된 토크나이징"
}

# 기본 라이브러리 사용 시 문제
from jamo import h2j, j2hcj

# "감"을 분리하면
h2j("감")  # 'ㄱㅏㅁ'

# 그런데 토크나이저가 이걸 어떻게 처리하느냐에 따라 성능 차이 발생
```

결국 자모 분리 로직을 직접 구현했습니다. 특히 받침 처리가 까다로웠습니다.

```python
class KoreanJamoProcessor:
    """
    한국어 자모 처리기

    핵심 개선:
    1. 겹받침 처리 로직 개선
    2. 음절 경계 인식 강화
    3. 토크나이저와의 정합성
    """

    # 초성, 중성, 종성 정의
    CHOSUNG = [
        'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ',
        'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
    ]
    JUNGSUNG = [
        'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
        'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
    ]
    JONGSUNG = [
        '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
        'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
        'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
    ]

    # 겹받침 분해 규칙
    DOUBLE_JONGSUNG = {
        'ㄳ': ('ㄱ', 'ㅅ'),
        'ㄵ': ('ㄴ', 'ㅈ'),
        'ㄶ': ('ㄴ', 'ㅎ'),
        'ㄺ': ('ㄹ', 'ㄱ'),
        'ㄻ': ('ㄹ', 'ㅁ'),
        'ㄼ': ('ㄹ', 'ㅂ'),
        'ㄽ': ('ㄹ', 'ㅅ'),
        'ㄾ': ('ㄹ', 'ㅌ'),
        'ㄿ': ('ㄹ', 'ㅍ'),
        'ㅀ': ('ㄹ', 'ㅎ'),
        'ㅄ': ('ㅂ', 'ㅅ'),
    }

    def decompose(self, text: str) -> list[str]:
        """
        한글 텍스트를 자모 시퀀스로 분해합니다.

        개선점:
        - 겹받침은 두 개의 토큰으로 분리
        - 음절 경계에 특수 토큰 추가 (선택적)
        - 영어/숫자는 그대로 유지
        """
        result = []

        for char in text:
            if '가' <= char <= '힣':
                # 한글 음절 분해
                code = ord(char) - ord('가')
                cho = code // (21 * 28)
                jung = (code % (21 * 28)) // 28
                jong = code % 28

                result.append(self.CHOSUNG[cho])
                result.append(self.JUNGSUNG[jung])

                if jong > 0:
                    jongsung = self.JONGSUNG[jong]
                    # 겹받침 분해
                    if jongsung in self.DOUBLE_JONGSUNG:
                        j1, j2 = self.DOUBLE_JONGSUNG[jongsung]
                        result.extend([j1, j2])
                    else:
                        result.append(jongsung)
            else:
                # 비한글 문자는 그대로
                result.append(char)

        return result

    def compose(self, jamo_list: list[str]) -> str:
        """
        자모 시퀀스를 한글 텍스트로 조합합니다.
        """
        # 구현 생략 - 분해의 역과정
        pass


# 적용 후 성능 변화
JAMO_IMPROVEMENT = {
    "before": {
        "CER": "4.8%",
        "common_errors": ["받침 혼동", "겹받침 오류", "연음 처리 실패"]
    },
    "after": {
        "CER": "3.1%",
        "improvement": "35% 오류 감소"
    }
}
```

### 2. 일본어 표기 통일 문제

일본어는 또 다른 문제가 있었습니다. 같은 발음이 히라가나, 가타카나, 한자로 표기될 수 있습니다.

```python
# 일본어 표기의 다양성
JAPANESE_WRITING_SYSTEMS = {
    "arigatou": {
        "hiragana": "ありがとう",
        "kanji": "有難う",
        "same_pronunciation": True
    },
    "coffee": {
        "katakana": "コーヒー",
        "hiragana": "こーひー",  # 가능하지만 잘 안 씀
        "note": "외래어는 보통 가타카나"
    }
}

# 문제: 학습 데이터마다 표기가 다름
# → 모델이 혼란스러워함
# → 같은 발음에 대해 다른 출력
```

저희는 히라가나 기준으로 통일하고, 후처리에서 한자/가타카나로 변환하는 방식을 택했습니다.

```python
import MeCab
from typing import Optional

class JapanesePostProcessor:
    """
    일본어 후처리기

    전략:
    1. STT 출력은 히라가나로 통일
    2. 형태소 분석으로 한자 후보 추출
    3. 문맥에 맞는 표기로 변환
    """

    def __init__(self):
        # MeCab 초기화 (형태소 분석기)
        self.mecab = MeCab.Tagger("-Ochasen")

        # 가타카나로 표기해야 하는 외래어 사전
        self.gairaigo_dict = self.load_gairaigo_dict()

    def process(self, hiragana_text: str) -> str:
        """
        히라가나 텍스트를 적절한 표기로 변환합니다.
        """
        # 형태소 분석
        parsed = self.mecab.parse(hiragana_text)

        result = []
        for line in parsed.split('\n'):
            if line == 'EOS' or line == '':
                continue

            parts = line.split('\t')
            surface = parts[0]  # 표층형
            reading = parts[1] if len(parts) > 1 else surface  # 읽기

            # 외래어는 가타카나로
            if self.is_gairaigo(surface, reading):
                result.append(self.to_katakana(surface))
            # 일반 단어는 한자 변환 시도
            else:
                converted = self.try_kanji_conversion(surface, reading)
                result.append(converted)

        return ''.join(result)

    def is_gairaigo(self, surface: str, reading: str) -> bool:
        """외래어 여부 판정"""
        # 사전 기반 + 규칙 기반 판정
        if reading in self.gairaigo_dict:
            return True
        # 장음이 포함된 경우 외래어 가능성 높음
        if 'ー' in surface or self.has_long_vowel_pattern(reading):
            return True
        return False

    def to_katakana(self, hiragana: str) -> str:
        """히라가나를 가타카나로 변환"""
        return ''.join(
            chr(ord(c) + 96) if 'ぁ' <= c <= 'ん' else c
            for c in hiragana
        )


# 적용 결과
JAPANESE_IMPROVEMENT = {
    "strategy": "히라가나 통일 + 후처리 변환",
    "before_CER": "5.1%",
    "after_CER": "3.8%",
    "readability": "후처리 덕분에 자연스러운 표기"
}
```

### 3. GPU 메모리 터짐

FastConformer가 생각보다 메모리를 많이 사용했습니다. Attention 연산이 시퀀스 길이의 제곱에 비례하기 때문에, 긴 오디오가 들어오면 OOM(Out of Memory)이 발생했습니다.

```python
# 메모리 사용량 분석
def estimate_memory(seq_len: int, d_model: int = 512, batch_size: int = 1):
    """
    Attention 메모리 사용량 추정

    Q, K, V 각각: batch_size * seq_len * d_model
    Attention matrix: batch_size * num_heads * seq_len * seq_len
    """
    qkv_memory = 3 * batch_size * seq_len * d_model * 4  # float32
    attn_memory = batch_size * 8 * seq_len * seq_len * 4  # 8 heads

    total_mb = (qkv_memory + attn_memory) / (1024 * 1024)
    return total_mb

# 시퀀스 길이별 메모리
MEMORY_BY_LENGTH = {
    "10sec (1000 frames)": f"{estimate_memory(1000):.1f} MB",
    "30sec (3000 frames)": f"{estimate_memory(3000):.1f} MB",
    "60sec (6000 frames)": f"{estimate_memory(6000):.1f} MB",
    "300sec (30000 frames)": f"{estimate_memory(30000):.1f} MB"  # OOM!
}
```

여러 가지 최적화 기법을 적용했습니다.

```python
import torch
from torch.cuda.amp import autocast, GradScaler

class OptimizedFastConformer:
    """
    메모리 최적화된 FastConformer 래퍼
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config

        # Mixed Precision 설정
        self.use_fp16 = config.get("use_fp16", True)

        # Gradient Checkpointing 활성화 (학습 시)
        if config.get("gradient_checkpointing", True):
            self.model.encoder.gradient_checkpointing_enable()

    def transcribe(self, audio: torch.Tensor) -> str:
        """
        최적화된 추론
        """
        with torch.no_grad():
            # FP16으로 메모리 절반 절약
            with autocast(enabled=self.use_fp16):
                # 청크 단위로 처리하여 피크 메모리 감소
                chunks = self.split_audio(audio, chunk_size=16000 * 30)  # 30초

                results = []
                for chunk in chunks:
                    result = self.model.transcribe([chunk])
                    results.append(result[0])

                    # 중간 결과 메모리 해제
                    torch.cuda.empty_cache()

        return ' '.join(results)

    def split_audio(self, audio: torch.Tensor, chunk_size: int) -> list:
        """
        긴 오디오를 청크로 분할
        오버랩을 두어 경계에서의 품질 저하 방지
        """
        overlap = chunk_size // 10  # 10% 오버랩
        chunks = []

        start = 0
        while start < len(audio):
            end = min(start + chunk_size, len(audio))
            chunks.append(audio[start:end])
            start = end - overlap

        return chunks


# 최적화 전후 비교
OPTIMIZATION_RESULTS = {
    "baseline": {
        "max_audio_length": "60초",
        "memory_usage": "12GB",
        "batch_size": 1
    },
    "optimized": {
        "max_audio_length": "무제한 (청크 처리)",
        "memory_usage": "4GB",
        "batch_size": 4,  # 같은 메모리로 4배 처리량
        "techniques": [
            "FP16 (메모리 50% 감소)",
            "Gradient Checkpointing (학습 시 40% 감소)",
            "청크 단위 처리 (피크 메모리 제어)",
            "중간 결과 캐시 해제"
        ]
    }
}
```

## 학습 파이프라인

사전학습된 모델을 한국어/일본어에 파인튜닝했습니다.

```python
from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf
import pytorch_lightning as pl

# 파인튜닝 설정
config = OmegaConf.create({
    "model": {
        "train_ds": {
            "manifest_filepath": "train_manifest.json",
            "batch_size": 16,
            "num_workers": 8,
            "pin_memory": True,
            "shuffle": True,
        },
        "validation_ds": {
            "manifest_filepath": "val_manifest.json",
            "batch_size": 16,
            "num_workers": 4,
        },
        "optim": {
            "name": "adamw",
            "lr": 1e-4,
            "weight_decay": 1e-3,
            "sched": {
                "name": "CosineAnnealing",
                "warmup_steps": 1000,
                "min_lr": 1e-6,
            }
        }
    },
    "trainer": {
        "devices": 4,  # 4 GPU
        "accelerator": "gpu",
        "strategy": "ddp",
        "max_epochs": 50,
        "accumulate_grad_batches": 2,
        "precision": 16,  # FP16 학습
        "gradient_clip_val": 1.0,
    }
})

# 모델 로드 및 파인튜닝
model = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-1.1b")

# 토크나이저 교체 (한국어/일본어용)
model.change_vocabulary(
    new_tokenizer_dir="./korean_japanese_tokenizer",
    new_tokenizer_type="bpe"
)

# 학습
trainer = pl.Trainer(**config.trainer)
trainer.fit(model)
```

데이터 준비가 가장 중요했습니다.

```python
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class AudioSample:
    audio_filepath: str
    text: str
    duration: float
    language: str  # "ko" or "ja"

def create_manifest(samples: list[AudioSample], output_path: str):
    """
    NeMo 형식의 manifest 파일 생성

    포맷:
    {"audio_filepath": "path/to/audio.wav", "text": "transcription", "duration": 3.5}
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            entry = {
                "audio_filepath": sample.audio_filepath,
                "text": sample.text,
                "duration": sample.duration
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# 데이터 품질 체크
def validate_dataset(manifest_path: str) -> dict:
    """
    데이터셋 품질 검증
    """
    issues = {
        "missing_audio": [],
        "too_short": [],
        "too_long": [],
        "empty_text": [],
        "duration_mismatch": []
    }

    with open(manifest_path, 'r') as f:
        for line in f:
            entry = json.loads(line)

            # 파일 존재 확인
            if not Path(entry["audio_filepath"]).exists():
                issues["missing_audio"].append(entry["audio_filepath"])
                continue

            # 너무 짧은 오디오 (1초 미만)
            if entry["duration"] < 1.0:
                issues["too_short"].append(entry["audio_filepath"])

            # 너무 긴 오디오 (30초 초과) - 청크로 나눠야 함
            if entry["duration"] > 30.0:
                issues["too_long"].append(entry["audio_filepath"])

            # 빈 텍스트
            if not entry["text"].strip():
                issues["empty_text"].append(entry["audio_filepath"])

    return issues


# 데이터 통계
DATASET_STATS = {
    "korean": {
        "total_hours": 5000,
        "sources": ["방송 자막", "유튜브", "팟캐스트", "콜센터"],
        "speakers": "다수 (10000+)"
    },
    "japanese": {
        "total_hours": 3000,
        "sources": ["NHK", "유튜브", "오디오북"],
        "speakers": "다수 (5000+)"
    }
}
```

## 결과

최종 성능입니다.

```python
# 평가 결과
FINAL_RESULTS = {
    "korean": {
        "CER": "3.1%",  # 기존 4.5% → 3.1%
        "improvement": "31% 개선",
        "test_set": "내부 테스트셋 50시간"
    },
    "japanese": {
        "CER": "3.8%",  # 기존 5.1% → 3.8%
        "improvement": "25% 개선",
        "test_set": "내부 테스트셋 30시간"
    },
    "rtf": {
        "value": 0.18,
        "meaning": "1초 오디오를 0.18초에 처리",
        "implication": "실시간보다 5.5배 빠름"
    },
    "latency": {
        "p50": "85ms",
        "p90": "120ms",
        "p99": "180ms"
    }
}

# RTF의 의미
"""
RTF(Real-Time Factor) = 처리 시간 / 오디오 시간

RTF < 1.0 → 실시간 처리 가능
RTF = 0.18 → 1시간 오디오를 11분에 처리

서비스 관점:
- 동시 처리 가능한 스트림 수 = 1 / RTF
- RTF 0.18이면 이론상 GPU 1개로 5.5개 스트림 동시 처리
"""
```

## 서빙 최적화

실제 서비스에 배포하기 위해 추가 최적화를 진행했습니다.

```python
import triton_python_backend_utils as pb_utils
import numpy as np

class TritonFastConformerModel:
    """
    NVIDIA Triton Inference Server용 모델 래퍼

    장점:
    - 동적 배칭: 여러 요청을 자동으로 묶어서 처리
    - 모델 앙상블: 전처리 → 추론 → 후처리 파이프라인
    - 멀티 인스턴스: GPU당 여러 모델 인스턴스 실행
    """

    def __init__(self):
        self.model = self.load_optimized_model()

    def load_optimized_model(self):
        """
        TensorRT 최적화된 모델 로드
        """
        import tensorrt as trt

        # ONNX → TensorRT 변환 시 최적화
        # - FP16 연산
        # - Layer fusion
        # - Kernel auto-tuning

        return trt_model

    def execute(self, requests):
        """
        Triton에서 호출하는 추론 함수
        """
        responses = []

        for request in requests:
            audio = pb_utils.get_input_tensor_by_name(request, "audio")
            audio_np = audio.as_numpy()

            # 추론
            with torch.no_grad():
                text = self.model.transcribe(audio_np)

            # 응답 생성
            output = pb_utils.Tensor("text", np.array([text]))
            response = pb_utils.InferenceResponse(output_tensors=[output])
            responses.append(response)

        return responses


# Triton 설정
TRITON_CONFIG = """
name: "fastconformer_korean"
platform: "python"
max_batch_size: 8

input [
  {
    name: "audio"
    data_type: TYPE_FP32
    dims: [-1]  # 가변 길이
  }
]

output [
  {
    name: "text"
    data_type: TYPE_STRING
    dims: [1]
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
  }
]

dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 100000
}
"""
```

## 배운 점

```python
LESSONS_LEARNED = {
    "preprocessing": {
        "lesson": "전처리가 절반이다",
        "detail": "모델 아키텍처보다 데이터 전처리가 성능에 더 큰 영향",
        "example": "자모 분리 로직 개선만으로 CER 1.4%p 개선"
    },
    "language_specific": {
        "lesson": "언어마다 다르다",
        "detail": "영어 논문 그대로 따라하면 안 된다",
        "example": "한국어는 자모, 일본어는 표기 통일이 핵심"
    },
    "realtime_challenge": {
        "lesson": "실시간이 어렵다",
        "detail": "배치 처리는 쉬운데 스트리밍은 완전 다른 문제",
        "example": "캐시 관리, 청크 경계 처리, 지연시간 최적화 필요"
    },
    "memory_management": {
        "lesson": "메모리는 항상 부족하다",
        "detail": "실서비스에서는 동시 요청 처리가 중요",
        "example": "FP16 + 청크 처리로 4배 처리량 확보"
    }
}
```

다음에는 화자 분리(Speaker Diarization)도 붙여볼 생각입니다. 회의록 자동 생성 같은 기능을 만들려면 "누가 말했는지"를 알아야 하니까요.

## 참고 자료

- [FastConformer Paper](https://arxiv.org/abs/2305.05084) - NVIDIA
- [NeMo Toolkit](https://github.com/NVIDIA/NeMo) - NVIDIA ASR 프레임워크
- [Parakeet Models](https://huggingface.co/nvidia/parakeet-ctc-1.1b) - 사전학습 모델
- [Triton Inference Server](https://github.com/triton-inference-server/server) - 서빙 프레임워크
- [한국어 음성인식 연구 동향](https://www.koreascience.or.kr/) - 국내 연구

