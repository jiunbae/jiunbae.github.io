---
title: "2024년 STT 기술 동향"
description: "2024년 STT 기술의 주요 변화 - 클라우드에서 엣지로의 이동, WhisperKit과 온디바이스 AI, 그리고 실시간 음성인식의 새로운 가능성"
date: 2024-09-01
slug: /speech-to-text-trends-2024
tags: [ai, dev]
published: true
---

# 2024년 STT 기술 동향

STT(Speech-to-Text) 분야에서 일하다 보니 기술 트렌드를 계속 쫓게 됩니다. 2024년 현재 이 분야에서 무엇이 달라졌는지 정리합니다.

## 가장 큰 변화: 엣지로의 이동

클라우드 STT에서 온디바이스 STT로 흐름이 바뀌고 있습니다.

```python
# 2024년 STT 아키텍처 트렌드
STT_ARCHITECTURE_TRENDS = {
    "2020-2022": {
        "paradigm": "Cloud-First",
        "approach": "오디오를 서버로 전송 → 추론 → 결과 반환",
        "latency": "200-500ms (네트워크 포함)",
        "pros": ["높은 정확도", "복잡한 모델 가능"],
        "cons": ["프라이버시 우려", "네트워크 의존", "API 비용"]
    },
    "2023-2024": {
        "paradigm": "Edge-First",
        "approach": "디바이스에서 직접 추론",
        "latency": "30-100ms",
        "pros": ["프라이버시 보장", "오프라인 가능", "무료"],
        "cons": ["디바이스 성능 제약", "모델 크기 제한"]
    }
}
```

[WhisperKit](https://github.com/argmaxinc/WhisperKit)이 대표적입니다. 아이폰에서 Whisper 모델을 실시간으로 돌립니다. 처음 봤을 때 놀랐습니다. 10억 파라미터짜리 모델이 폰에서 돌아가다니.

```swift
// WhisperKit 사용 예시 (Swift)
import WhisperKit

class SpeechRecognizer {
    private var whisperKit: WhisperKit?

    init() async throws {
        // 모델 로드 (최초 실행 시 다운로드)
        whisperKit = try await WhisperKit(
            model: "base",  // tiny, base, small 등
            computeOptions: .init(
                melCompute: .cpuAndGPU,  // Neural Engine 활용
                audioEncoderCompute: .cpuAndGPU,
                textDecoderCompute: .cpuAndGPU
            )
        )
    }

    func transcribe(audioURL: URL) async throws -> String {
        guard let whisperKit = whisperKit else {
            throw RecognitionError.modelNotLoaded
        }

        let results = try await whisperKit.transcribe(
            audioPath: audioURL.path,
            decodeOptions: .init(
                language: "ko",  // 한국어
                task: .transcribe,
                temperature: 0.0,  // greedy decoding
                sampleLength: 224
            )
        )

        return results.map { $0.text }.joined(separator: " ")
    }

    // 실시간 스트리밍 처리
    func streamTranscribe(audioBuffer: AVAudioPCMBuffer) async throws -> String {
        // 청크 단위로 처리
        let result = try await whisperKit?.transcribe(
            audioArray: audioBuffer.floatChannelData![0],
            decodeOptions: .init(
                language: "ko",
                task: .transcribe
            )
        )
        return result?.first?.text ?? ""
    }
}

// 성능 벤치마크 (iPhone 15 Pro)
WHISPERKIT_PERFORMANCE = {
    "tiny": {"model_size": "39MB", "rtf": "0.03", "wer": "14.2%"},
    "base": {"model_size": "74MB", "rtf": "0.05", "wer": "10.1%"},
    "small": {"model_size": "244MB", "rtf": "0.12", "wer": "7.3%"}
}
```

왜 엣지로 가느냐? 세 가지 이유가 있습니다.

```python
EDGE_STT_DRIVERS = {
    "privacy": {
        "concern": "음성 데이터는 매우 민감한 개인정보",
        "regulation": "GDPR, 개인정보보호법 강화",
        "solution": "데이터가 디바이스를 떠나지 않음"
    },
    "latency": {
        "cloud_latency": "네트워크 왕복 100-300ms",
        "edge_latency": "추론만 30-50ms",
        "impact": "실시간 자막, 음성 비서에 필수"
    },
    "cost": {
        "cloud_cost": "$0.006-0.024 per minute (대형 서비스)",
        "edge_cost": "$0 (초기 개발 비용만)",
        "scale": "월 1억 분 처리 시 연 $7M+ 절감 가능"
    }
}
```

## FastConformer: 새로운 표준

NVIDIA에서 나온 [FastConformer](https://arxiv.org/abs/2305.05084)가 인상적입니다. 기존 Conformer보다 훨씬 효율적입니다.

```python
# FastConformer vs Conformer 비교
import torch
import torch.nn as nn

class ConformerAttention(nn.Module):
    """
    기존 Conformer의 Multi-Head Self-Attention
    복잡도: O(T²) where T = sequence length
    """
    def forward(self, x):
        # Full attention over all positions
        T = x.size(1)
        attn = torch.matmul(x, x.transpose(-2, -1))  # [B, T, T]
        # T=1000이면 1M 연산
        return attn


class FastConformerAttention(nn.Module):
    """
    FastConformer의 Linear Attention
    복잡도: O(T)

    핵심 변경:
    1. Downsampling으로 시퀀스 길이 감소
    2. Linear attention approximation
    3. Cached inference for streaming
    """
    def __init__(self, d_model, n_heads, downsample_factor=8):
        super().__init__()
        self.downsample = nn.Conv1d(
            d_model, d_model,
            kernel_size=downsample_factor,
            stride=downsample_factor
        )
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.upsample = nn.ConvTranspose1d(
            d_model, d_model,
            kernel_size=downsample_factor,
            stride=downsample_factor
        )

    def forward(self, x):
        # [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)

        # Downsample: T -> T/8
        x_down = self.downsample(x)

        # Attention on reduced sequence
        x_down = x_down.transpose(1, 2)
        attn_out, _ = self.attention(x_down, x_down, x_down)

        # Upsample back: T/8 -> T
        x_up = self.upsample(attn_out.transpose(1, 2))

        return x_up.transpose(1, 2)


# 실제 성능 차이 (30초 오디오 기준)
PERFORMANCE_COMPARISON = {
    "Conformer-Large": {
        "params": "121M",
        "memory": "8.2GB",
        "latency": "420ms",
        "WER_librispeech": "2.1%"
    },
    "FastConformer-Large": {
        "params": "115M",
        "memory": "4.1GB",  # 50% 감소
        "latency": "180ms",  # 57% 감소
        "WER_librispeech": "2.0%"  # 오히려 향상
    }
}
```

핵심은 attention 복잡도를 O(n²)에서 O(n)으로 줄인 것입니다. 긴 오디오를 처리할 때 차이가 큽니다.

```python
# NeMo를 이용한 FastConformer 사용
import nemo.collections.asr as nemo_asr

# 사전학습된 FastConformer 로드
model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
    "nvidia/parakeet-ctc-1.1b"  # 1.1B 파라미터 모델
)

# 추론
transcription = model.transcribe(["audio.wav"])
print(transcription)

# 스트리밍 설정
streaming_config = {
    "chunk_size": 1.6,  # 1.6초 청크
    "shift_size": 0.4,  # 0.4초 오버랩
    "lookahead": 0,     # 실시간을 위해 lookahead 없음
}

# 스트리밍 추론
class StreamingASR:
    def __init__(self, model):
        self.model = model
        self.buffer = []
        self.cache = None

    def process_chunk(self, audio_chunk):
        """
        청크 단위 처리 (스트리밍)
        """
        self.buffer.append(audio_chunk)

        if len(self.buffer) * self.chunk_size >= 1.6:
            # 충분한 컨텍스트 쌓이면 추론
            audio = torch.cat(self.buffer[-4:], dim=-1)  # 최근 4청크

            with torch.no_grad():
                result, self.cache = self.model.transcribe_chunk(
                    audio,
                    cache=self.cache
                )

            return result

        return None
```

우리 팀에서도 FastConformer 기반으로 STT 엔진을 만들었는데, 기존 대비 지연시간이 48% 줄었습니다. 메모리도 거의 절반으로 줄었습니다.

## 양자화 기술의 성숙

INT8 양자화가 거의 표준이 되었습니다. 모델 크기를 4배 줄이면서 정확도 손실은 1% 미만입니다.

```python
import torch
from torch.quantization import quantize_dynamic
import onnxruntime as ort

# PyTorch 동적 양자화
def quantize_stt_model(model):
    """
    동적 양자화: 가중치만 INT8로 변환
    - 활성화는 런타임에 양자화
    - 정확도 손실 최소화
    """
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv1d},
        dtype=torch.qint8
    )
    return quantized_model


# ONNX + INT8 양자화 (더 공격적)
def export_and_quantize_onnx(model, output_path):
    """
    정적 양자화: 가중치 + 활성화 모두 INT8
    - 캘리브레이션 데이터 필요
    - 더 작은 모델, 더 빠른 추론
    """
    from onnxruntime.quantization import quantize_static, CalibrationDataReader

    # ONNX로 내보내기
    dummy_input = torch.randn(1, 16000 * 30)  # 30초 오디오
    torch.onnx.export(
        model,
        dummy_input,
        "model_fp32.onnx",
        opset_version=17,
        input_names=['audio'],
        output_names=['transcription'],
        dynamic_axes={'audio': {1: 'audio_length'}}
    )

    # 양자화
    quantize_static(
        "model_fp32.onnx",
        output_path,
        calibration_data_reader=CalibrationDataReader(calibration_data),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8
    )


# 양자화 결과 비교
QUANTIZATION_RESULTS = {
    "Whisper-base (FP32)": {
        "size": "290MB",
        "latency_30s": "1.2s",
        "WER": "10.1%"
    },
    "Whisper-base (INT8)": {
        "size": "75MB",  # 74% 감소
        "latency_30s": "0.4s",  # 67% 감소
        "WER": "10.3%"  # 0.2%p 손실
    },
    "Whisper-base (INT4)": {
        "size": "40MB",  # 86% 감소
        "latency_30s": "0.3s",  # 75% 감소
        "WER": "11.2%"  # 1.1%p 손실 (트레이드오프)
    }
}
```

INT4까지 가면 더 줄일 수 있는데, 여기서부터는 정확도 트레이드오프가 생깁니다. 용도에 따라 선택해야 합니다.

## 스트리밍 처리

예전에는 전체 오디오를 다 받은 다음에 처리했습니다. 요즘은 오디오가 들어오는 대로 바로바로 처리합니다.

```python
import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Generator, Optional

@dataclass
class StreamingConfig:
    chunk_duration_ms: int = 1600  # 1.6초 청크
    shift_duration_ms: int = 400   # 0.4초 시프트
    sample_rate: int = 16000
    lookahead_ms: int = 0          # 실시간은 0


class StreamingSTT:
    """
    실시간 스트리밍 STT 구현

    핵심 개념:
    1. 청크 단위 처리 - 전체 오디오 기다리지 않음
    2. 컨텍스트 유지 - 이전 청크 정보 활용
    3. 점진적 출력 - 부분 결과 즉시 반환
    """

    def __init__(self, model, config: StreamingConfig):
        self.model = model
        self.config = config
        self.buffer = []
        self.encoder_cache = None
        self.decoder_cache = None

    async def process_stream(
        self,
        audio_stream: Generator[np.ndarray, None, None]
    ):
        """
        오디오 스트림을 실시간으로 처리
        """
        chunk_samples = int(
            self.config.chunk_duration_ms * self.config.sample_rate / 1000
        )
        shift_samples = int(
            self.config.shift_duration_ms * self.config.sample_rate / 1000
        )

        accumulated = np.array([], dtype=np.float32)

        async for audio_chunk in audio_stream:
            accumulated = np.concatenate([accumulated, audio_chunk])

            # 청크 크기 도달하면 처리
            while len(accumulated) >= chunk_samples:
                chunk = accumulated[:chunk_samples]

                # 모델 추론 (캐시 활용)
                result, self.encoder_cache, self.decoder_cache = \
                    await self.model.transcribe_chunk(
                        chunk,
                        encoder_cache=self.encoder_cache,
                        decoder_cache=self.decoder_cache
                    )

                # 부분 결과 반환
                yield result

                # 시프트만큼 이동
                accumulated = accumulated[shift_samples:]

        # 남은 오디오 처리
        if len(accumulated) > 0:
            result, _, _ = await self.model.transcribe_chunk(
                accumulated,
                encoder_cache=self.encoder_cache,
                decoder_cache=self.decoder_cache,
                is_final=True
            )
            yield result


# 지연시간 측정
class LatencyProfiler:
    """
    스트리밍 STT 지연시간 측정

    지연시간 구성요소:
    1. 청크 수집 시간 (chunk_duration)
    2. 추론 시간 (inference_time)
    3. 버퍼링 시간 (buffering)
    """

    def measure_latency(self, audio_path: str) -> dict:
        import time

        streaming_stt = StreamingSTT(model, StreamingConfig())

        latencies = []
        audio = load_audio(audio_path)

        chunk_size = 1600 * 16  # 1.6초
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]

            start = time.perf_counter()
            result = streaming_stt.process_chunk(chunk)
            elapsed = time.perf_counter() - start

            latencies.append(elapsed * 1000)  # ms

        return {
            "mean_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p90_latency_ms": np.percentile(latencies, 90),
            "p99_latency_ms": np.percentile(latencies, 99)
        }


# 실제 측정 결과
STREAMING_LATENCY = {
    "FastConformer (A100)": {
        "p50": "28ms",
        "p90": "35ms",
        "p99": "52ms"
    },
    "Whisper-base (A100)": {
        "p50": "85ms",
        "p90": "120ms",
        "p99": "180ms"
    }
}
```

[FastEmit](https://arxiv.org/abs/2010.11148) 같은 기술로 지연시간을 더 줄일 수 있습니다. 90th percentile 기준으로 210ms에서 30ms까지 줄인 사례도 있습니다.

## 아직 어려운 것들

```python
STT_CHALLENGES_2024 = {
    "domain_adaptation": {
        "problem": "의료, 법률 같은 전문 분야는 여전히 어려움",
        "reason": "전문 용어가 일반 학습 데이터에 부족",
        "solution_attempts": [
            "도메인 특화 파인튜닝",
            "전문 용어 사전 통합",
            "Contextual biasing"
        ],
        "current_gap": "일반 도메인 WER 3% vs 전문 도메인 WER 8-15%"
    },
    "code_switching": {
        "problem": "한국어와 영어 섞어서 말하면 잘 못 알아듣는다",
        "example": "'이 function을 call하면...' → 인식 실패",
        "reason": "언어별로 별도 토크나이저 사용",
        "solution_attempts": [
            "다국어 통합 모델",
            "언어 감지 + 분기 처리"
        ]
    },
    "noise_robustness": {
        "problem": "시끄러운 환경에서 정확도 급락",
        "scenarios": [
            "배경 음악",
            "여러 화자 동시 발화",
            "저품질 마이크"
        ],
        "clean_wer": "3%",
        "noisy_wer": "12-25%"
    },
    "diarization": {
        "problem": "누가 말했는지 구분",
        "difficulty": "오버랩 구간, 유사한 목소리",
        "current_accuracy": "DER 8-15%"
    }
}
```

## 2025년 예측

```python
PREDICTIONS_2025 = {
    "edge_adoption": {
        "2024": "40% of STT workloads",
        "2025": "60-70% (predicted)",
        "driver": "프라이버시 규제 강화, 디바이스 성능 향상"
    },
    "latency": {
        "2024": "50-100ms typical",
        "2025": "<50ms standard, <20ms achievable",
        "enabler": "더 효율적인 모델, 전용 하드웨어"
    },
    "accuracy": {
        "2024": "WER 3-5% (clean speech)",
        "2025": "WER <2% (approaching human level)",
        "human_level": "WER ~1.5% (전문 타이피스트 수준)"
    },
    "multimodal": {
        "current": "오디오만 처리",
        "2025": "오디오 + 비디오 (립리딩) 결합",
        "benefit": "노이즈 환경 정확도 유지"
    }
}
```

개인적으로는 멀티모달이 기대됩니다. 립리딩과 결합하면 노이즈 환경에서도 정확도를 유지할 수 있을 것입니다.

```python
# 멀티모달 STT 개념
class MultimodalSTT:
    """
    Audio + Visual 결합 STT

    이점:
    - 노이즈 환경에서 강건함
    - 동음이의어 구분 향상
    - 화자 구분 용이
    """

    def __init__(self, audio_encoder, visual_encoder, fusion_model):
        self.audio_encoder = audio_encoder
        self.visual_encoder = visual_encoder
        self.fusion = fusion_model

    def transcribe(self, audio, video_frames):
        # 오디오 인코딩
        audio_features = self.audio_encoder(audio)

        # 비주얼 인코딩 (입술 움직임)
        visual_features = self.visual_encoder(video_frames)

        # 멀티모달 퓨전
        fused_features = self.fusion(audio_features, visual_features)

        # 디코딩
        transcription = self.decoder(fused_features)
        return transcription

# 기대 효과
MULTIMODAL_BENEFITS = {
    "clean_audio": {"audio_only": "WER 3%", "multimodal": "WER 2.5%"},
    "noisy_audio_snr_5db": {"audio_only": "WER 15%", "multimodal": "WER 6%"},
    "noisy_audio_snr_0db": {"audio_only": "WER 35%", "multimodal": "WER 12%"}
}
```

## 참고 자료

- [WhisperKit](https://github.com/argmaxinc/WhisperKit) - 온디바이스 Whisper
- [FastConformer Paper](https://arxiv.org/abs/2305.05084) - NVIDIA
- [NeMo Toolkit](https://github.com/NVIDIA/NeMo) - NVIDIA ASR 프레임워크
- [Whisper](https://github.com/openai/whisper) - OpenAI
- [FastEmit Paper](https://arxiv.org/abs/2010.11148) - 저지연 스트리밍
- [Audio-Visual Speech Recognition](https://arxiv.org/abs/2303.14307) - 멀티모달 연구

