---
title: "멀티모달 페이셜 애니메이션 시스템 개발기"
description: "NCSOFT에서 디지털 휴먼의 얼굴 애니메이션 시스템을 만들면서 배운 것들"
date: 2023-02-01
slug: /digital-human-facial-animation
tags: [ai, dev]
published: true
---

# 멀티모달 페이셜 애니메이션 시스템 개발기

NCSOFT Graphics AI Lab에서 디지털 휴먼의 얼굴 애니메이션 시스템을 개발했습니다. 오디오, 텍스트, 감정 정보를 입력받아 자연스러운 표정과 립싱크를 생성하는 시스템이었습니다. 이 글에서는 그 과정에서 겪은 기술적 도전과 해결 방법을 공유하고자 합니다.

## 왜 멀티모달인가

처음에는 오디오만으로 립싱크를 생성하는 모델을 만들었습니다. 기술적으로는 성공했지만, 결과물이 뭔가 어색했습니다.

문제를 분석해보니, 말하는 내용과 표정이 맞지 않는 경우가 많았습니다. 슬픈 내용을 말하는데 표정은 무표정하다든지, 농담을 하는데 진지한 표정이라든지.

```python
# 초기 Audio-only 모델 구조
class AudioToFaceModel(nn.Module):
    def __init__(self, audio_dim=80, hidden_dim=256, output_dim=52):
        super().__init__()
        self.audio_encoder = AudioEncoder(audio_dim, hidden_dim)
        self.face_decoder = FaceDecoder(hidden_dim, output_dim)

    def forward(self, mel_spectrogram):
        audio_features = self.audio_encoder(mel_spectrogram)
        blendshapes = self.face_decoder(audio_features)
        return blendshapes  # 52개의 ARKit blendshape 가중치
```

이 구조의 한계는 명확했습니다. 오디오만으로는 "무슨 말을 하는지"의 맥락을 파악하기 어려웠습니다. 같은 억양이라도 상황에 따라 다른 표정이 자연스러울 수 있는데, 그 정보가 오디오에는 없었습니다.

그래서 텍스트와 감정 정보를 함께 활용하기로 했습니다.

```python
# 멀티모달 모델 구조
class MultimodalFaceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 세 가지 모달리티의 인코더
        self.audio_encoder = AudioEncoder(config.audio_dim, config.hidden_dim)
        self.text_encoder = TextEncoder(config.vocab_size, config.hidden_dim)
        self.emotion_encoder = EmotionEncoder(config.num_emotions, config.hidden_dim)

        # 멀티모달 퓨전
        self.fusion = CrossModalFusion(config.hidden_dim, num_heads=8)

        # 얼굴 애니메이션 디코더
        self.face_decoder = TemporalFaceDecoder(config.hidden_dim, config.output_dim)

    def forward(self, audio, text, emotion):
        # 각 모달리티 인코딩
        audio_feat = self.audio_encoder(audio)      # [B, T, H]
        text_feat = self.text_encoder(text)          # [B, S, H]
        emotion_feat = self.emotion_encoder(emotion) # [B, H]

        # Cross-attention 기반 퓨전
        fused_feat = self.fusion(audio_feat, text_feat, emotion_feat)

        # 시간축을 고려한 얼굴 애니메이션 생성
        blendshapes = self.face_decoder(fused_feat)
        return blendshapes  # [B, T, 52]
```

## 100ms의 벽: 실시간 성능 달성

게임에서 사용하려면 실시간 처리가 필수였습니다. 목표는 100ms 이하의 지연시간이었습니다.

첫 버전의 성능 측정 결과:

```
처리 시간 분석 (1초 음성 기준):
- Audio 전처리: 45ms
- Audio Encoder: 85ms
- Text Encoder: 35ms
- Emotion Encoder: 5ms
- Fusion: 60ms
- Face Decoder: 50ms
총: 280ms
```

280ms는 너무 느렸습니다. 여러 최적화를 적용했습니다.

### 1. 모델 경량화

```python
# Knowledge Distillation
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_output, teacher_output, target):
        # Teacher의 soft label과 실제 target을 모두 활용
        soft_loss = self.kl_div(
            F.log_softmax(student_output / self.temperature, dim=-1),
            F.softmax(teacher_output / self.temperature, dim=-1)
        ) * (self.temperature ** 2)

        hard_loss = F.mse_loss(student_output, target)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

# Teacher 모델 (큰 모델)에서 Student 모델 (작은 모델)로 지식 전달
def train_with_distillation(teacher, student, dataloader):
    teacher.eval()
    student.train()

    for batch in dataloader:
        with torch.no_grad():
            teacher_output = teacher(batch)

        student_output = student(batch)
        loss = distillation_loss(student_output, teacher_output, batch['target'])

        loss.backward()
        optimizer.step()
```

### 2. 양자화

```python
import torch.quantization as quant

# 동적 양자화 적용
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM},
    dtype=torch.qint8
)

# 정적 양자화를 위한 준비
model.qconfig = quant.get_default_qconfig('fbgemm')
model_prepared = quant.prepare(model)

# 캘리브레이션 (대표 데이터로)
for batch in calibration_loader:
    model_prepared(batch)

# 양자화 적용
model_quantized = quant.convert(model_prepared)
```

### 3. TensorRT 변환

```python
import torch.onnx
import tensorrt as trt

# 1. PyTorch → ONNX
def export_to_onnx(model, sample_input, output_path):
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        export_params=True,
        opset_version=13,
        input_names=['audio', 'text', 'emotion'],
        output_names=['blendshapes'],
        dynamic_axes={
            'audio': {0: 'batch', 1: 'time'},
            'blendshapes': {0: 'batch', 1: 'time'}
        }
    )

# 2. ONNX → TensorRT
def build_tensorrt_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)  # FP16 활성화
    config.max_workspace_size = 1 << 30    # 1GB

    engine = builder.build_engine(network, config)

    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
```

최적화 결과:

```
최적화 후 처리 시간:
- Audio 전처리: 15ms (스트리밍 처리)
- TensorRT 추론: 65ms
- 후처리: 5ms
총: 85ms

95th percentile: 120ms
```

목표인 100ms를 대부분의 경우에서 달성했습니다.

## 페르소나: 캐릭터별 표현 스타일

같은 "기쁨" 감정이라도 캐릭터마다 표현 방식이 달라야 했습니다. 프로페셔널한 캐릭터는 절제된 미소를, 친근한 캐릭터는 활발한 웃음을 짓는 것이 자연스럽습니다.

이를 위해 페르소나 임베딩을 도입했습니다.

```python
class PersonaEmbedding(nn.Module):
    def __init__(self, num_personas, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_personas, embedding_dim)

        # 페르소나별 스타일 변환 파라미터
        self.style_transform = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, persona_id):
        emb = self.embedding(persona_id)
        style = self.style_transform(emb)
        return style


class PersonaAwareFaceDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_personas):
        super().__init__()
        self.persona_emb = PersonaEmbedding(num_personas, hidden_dim)
        self.base_decoder = TemporalFaceDecoder(hidden_dim, output_dim)

        # FiLM (Feature-wise Linear Modulation) 레이어
        self.gamma = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, features, persona_id):
        persona_style = self.persona_emb(persona_id)

        # 페르소나 스타일로 피처 변조
        gamma = self.gamma(persona_style).unsqueeze(1)  # [B, 1, H]
        beta = self.beta(persona_style).unsqueeze(1)

        modulated = gamma * features + beta

        return self.base_decoder(modulated)
```

페르소나 정의 예시:

```yaml
personas:
  professional:
    id: 0
    expression_range: 0.6  # 표현 범위 제한
    smile_intensity: 0.4
    blink_frequency: normal

  friendly:
    id: 1
    expression_range: 1.0  # 풀 표현
    smile_intensity: 0.8
    blink_frequency: frequent

  calm:
    id: 2
    expression_range: 0.5
    smile_intensity: 0.3
    blink_frequency: slow
```

## ARKit Blendshape 직접 예측

출력 형식은 ARKit의 52개 blendshape 가중치를 선택했습니다. 업계 표준이라 다양한 렌더러와 호환성이 좋았기 때문입니다.

```python
# ARKit Blendshape 목록 (일부)
ARKIT_BLENDSHAPES = [
    'browDownLeft', 'browDownRight', 'browInnerUp',
    'browOuterUpLeft', 'browOuterUpRight',
    'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookInLeft', 'eyeLookInRight',
    'eyeLookOutLeft', 'eyeLookOutRight',
    'eyeLookUpLeft', 'eyeLookUpRight',
    'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight',
    'jawForward', 'jawLeft', 'jawRight', 'jawOpen',
    'mouthClose', 'mouthFunnel', 'mouthPucker',
    'mouthLeft', 'mouthRight',
    'mouthSmileLeft', 'mouthSmileRight',
    # ... 총 52개
]

class BlendshapeDecoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.output = nn.Linear(hidden_dim, 52)
        self.sigmoid = nn.Sigmoid()  # blendshape는 0~1 범위

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        blendshapes = self.sigmoid(self.output(lstm_out))
        return blendshapes
```

처음에는 중간 표현(예: FLAME 파라미터)을 거쳐서 blendshape로 변환하는 방식을 시도했습니다. 하지만 직접 blendshape를 예측하는 것이 더 빠르고 정확했습니다.

## 평가와 결과

### 정량적 평가

```python
# 립싱크 정확도 평가
def evaluate_lip_sync(predicted, ground_truth, audio):
    """
    Lip Sync 정확도 측정
    - LMD (Lip Movement Distance): 예측과 GT의 입술 움직임 차이
    - LSE (Lip Sync Error): 오디오와 입술 움직임의 동기화 오류
    """
    # 입술 관련 blendshape만 추출
    lip_indices = [23, 24, 25, 26, 27, 28, 29, ...]  # jawOpen, mouthClose 등

    pred_lip = predicted[:, :, lip_indices]
    gt_lip = ground_truth[:, :, lip_indices]

    lmd = F.mse_loss(pred_lip, gt_lip).item()

    # 오디오-립 동기화 평가 (프레임 단위)
    lse = compute_sync_error(pred_lip, audio)

    return {'LMD': lmd, 'LSE': lse}
```

결과:

| 메트릭 | 우리 모델 | Baseline (Audio-only) |
|--------|----------|----------------------|
| LMD | 0.032 | 0.058 |
| LSE | 1.2 frames | 2.1 frames |
| 추론 시간 | 85ms | 65ms |

### 사용자 평가

고객 서비스 시나리오와 교육 시나리오에서 사용자 평가를 진행했습니다.

```
고객 서비스 시나리오:
- 전반적 만족도: 4.6/5.0
- 표정 자연스러움: 4.4/5.0
- 립싱크 정확도: 4.5/5.0

교육 시나리오:
- 전반적 만족도: 4.7/5.0
- 표정 자연스러움: 4.6/5.0
- 집중도: 4.3/5.0
```

교육 시나리오에서 더 좋은 결과가 나왔는데, "친근한" 페르소나가 잘 맞았던 것 같습니다.

## 어려웠던 점과 해결책

### 1. 오디오-텍스트 동기화

오디오와 텍스트의 타이밍이 맞지 않는 경우가 있었습니다. TTS로 생성된 오디오는 괜찮았지만, 실제 녹음 음성은 타이밍이 들쭉날쭉했습니다.

```python
# 강제 정렬 (Forced Alignment)로 해결
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def get_word_timestamps(audio, text):
    """
    Wav2Vec2를 활용한 단어 단위 타임스탬프 추출
    """
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    # CTC 디코딩으로 타임스탬프 추출
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    # 단어별 타임스탬프 계산
    timestamps = extract_word_timestamps(logits, text)
    return timestamps
```

### 2. 긴 대화에서의 일관성

긴 대화에서 페르소나 스타일이 흔들리는 문제가 있었습니다. 정규화 기법을 추가해서 해결했습니다.

```python
class ConsistencyRegularizer(nn.Module):
    def __init__(self, window_size=30):  # 30프레임 = 1초
        super().__init__()
        self.window_size = window_size

    def forward(self, blendshapes):
        """
        시간적 일관성을 위한 정규화
        급격한 변화에 페널티 부여
        """
        # 인접 프레임 간 차이
        temporal_diff = blendshapes[:, 1:, :] - blendshapes[:, :-1, :]

        # 급격한 변화 감지
        jerk = temporal_diff[:, 1:, :] - temporal_diff[:, :-1, :]

        # Smooth L1 loss로 급격한 변화에만 페널티
        consistency_loss = F.smooth_l1_loss(jerk, torch.zeros_like(jerk))

        return consistency_loss
```

### 3. 실시간 처리를 위한 버퍼링

스트리밍 입력을 처리하기 위해 버퍼링 시스템을 구현했습니다.

```python
class StreamingFaceAnimator:
    def __init__(self, model, buffer_size=0.5):  # 500ms 버퍼
        self.model = model
        self.buffer_size = buffer_size
        self.audio_buffer = []
        self.text_buffer = ""

    def process_chunk(self, audio_chunk, text_chunk=None):
        """
        청크 단위로 오디오를 처리하고 애니메이션 생성
        """
        self.audio_buffer.append(audio_chunk)

        if text_chunk:
            self.text_buffer += text_chunk

        # 버퍼가 충분히 쌓이면 처리
        if len(self.audio_buffer) * 0.02 >= self.buffer_size:  # 20ms 청크 기준
            audio = np.concatenate(self.audio_buffer)
            blendshapes = self.model.inference(audio, self.text_buffer)

            # 버퍼 클리어 (오버랩 유지)
            overlap = int(0.1 / 0.02)  # 100ms 오버랩
            self.audio_buffer = self.audio_buffer[-overlap:]

            return blendshapes

        return None
```

## 배운 것들

### 멀티모달의 효과

단일 모달리티보다 멀티모달 접근이 확실히 효과적이었습니다. 오디오만으로는 놓치는 맥락 정보를 텍스트와 감정 태그로 보완할 수 있었습니다.

### 실시간 제약의 중요성

모델 설계 초기부터 실시간 성능을 고려해야 합니다. 나중에 최적화하려면 훨씬 어렵습니다. 프로토타입 단계에서도 추론 시간을 측정하면서 개발하는 것이 좋습니다.

### 아티스트 피드백의 가치

기술적 메트릭만으로는 부족했습니다. 실제 애니메이터들의 피드백을 받으면서 많은 것을 개선할 수 있었습니다. 특히 "이 표정은 이 상황에서 어색하다" 같은 정성적 피드백이 중요했습니다.

## 참고 자료

- [Audio-Driven Facial Animation by Joint End-to-End Learning of Pose and Emotion](https://research.nvidia.com/publication/2017-07_Audio-Driven-Facial-Animation) - NVIDIA
- [ARKit Face Tracking](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation) - Apple Developer
- [VOCA: Capture, Learning, and Synthesis of 3D Speaking Styles](https://voca.is.tue.mpg.de/) - MPI
