---
title: "모바일에서 AI 모델 돌리기"
description: "스마트폰에서 AI 모델 돌리기 - Core ML, TensorFlow Lite 활용법과 모델 경량화, 양자화 기법까지 실무에서 배운 On-device AI 최적화 노하우"
date: 2023-06-01
slug: /mobile-ai-optimization-2023
tags: [ai, dev]
published: true
---

# 모바일에서 AI 모델 돌리기

2023년, On-device AI 프로젝트를 여러 개 진행하면서 많은 것을 배웠습니다. 스마트폰에서 AI 모델을 직접 돌리는 것이 생각보다 실용적인 수준에 도달했다는 것을 체감한 한 해였습니다. 이 글에서는 그 경험과 기술적인 내용을 정리합니다.

## 왜 모바일에서 돌리는가

On-device AI를 선택하는 이유는 크게 세 가지입니다.

```python
ON_DEVICE_AI_BENEFITS = {
    "privacy": {
        "description": "데이터를 서버로 전송하지 않음",
        "use_cases": ["음성 인식", "얼굴 인식", "건강 데이터 분석"],
        "compliance": ["GDPR", "HIPAA", "개인정보보호법"]
    },
    "latency": {
        "description": "네트워크 왕복 시간 제거",
        "server_api": "100-500ms (네트워크 포함)",
        "on_device": "10-50ms (추론만)",
        "improvement": "5-10x 빠름"
    },
    "cost": {
        "description": "API 호출 비용 없음",
        "api_cost_per_1k": "$0.001-0.01",
        "on_device_cost": "$0 (초기 개발 비용만)",
        "break_even": "대량 처리 시 유리"
    }
}
```

실제 프로젝트에서는 프라이버시가 가장 큰 동기였습니다. 음성 데이터나 카메라 피드를 외부 서버로 보내는 것에 대한 사용자 우려가 컸고, 로컬 처리로 이 문제를 해결할 수 있었습니다.

## 실제로 구현한 기능들

### 1. 이미지 분류

MobileNet 계열 모델을 활용한 이미지 분류를 구현했습니다. Core ML로 변환하면 iPhone에서 실시간으로 동작합니다.

```python
import coremltools as ct
import torch
from torchvision import models

# PyTorch MobileNetV3 로드
model = models.mobilenet_v3_small(pretrained=True)
model.eval()

# 입력 예시 생성
example_input = torch.rand(1, 3, 224, 224)

# TorchScript로 변환
traced_model = torch.jit.trace(model, example_input)

# Core ML 변환
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.ImageType(
        name="image",
        shape=(1, 3, 224, 224),
        scale=1/255.0,
        bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225]
    )],
    classifier_config=ct.ClassifierConfig("imagenet_labels.txt"),
    minimum_deployment_target=ct.target.iOS15
)

# Neural Engine 최적화 설정
coreml_model = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 224, 224))],
    compute_units=ct.ComputeUnit.ALL  # CPU, GPU, Neural Engine 모두 활용
)

coreml_model.save("MobileNetV3.mlpackage")
```

Swift에서 사용하는 코드는 다음과 같습니다.

```swift
import CoreML
import Vision

class ImageClassifier {
    private let model: VNCoreMLModel

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Neural Engine 활용

        let coreMLModel = try MobileNetV3(configuration: config)
        model = try VNCoreMLModel(for: coreMLModel.model)
    }

    func classify(image: CGImage) async throws -> [(label: String, confidence: Float)] {
        return try await withCheckedThrowingContinuation { continuation in
            let request = VNCoreMLRequest(model: model) { request, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }

                guard let results = request.results as? [VNClassificationObservation] else {
                    continuation.resume(returning: [])
                    return
                }

                let predictions = results.prefix(5).map {
                    (label: $0.identifier, confidence: $0.confidence)
                }
                continuation.resume(returning: predictions)
            }

            let handler = VNImageRequestHandler(cgImage: image)
            try? handler.perform([request])
        }
    }
}
```

### 2. 음성 인식 (Whisper)

Whisper 모델을 Core ML로 변환해서 iPhone에서 돌려봤습니다. tiny와 base 모델이 모바일에서 실용적인 수준으로 동작했습니다.

```python
import whisper
import coremltools as ct
import torch

# Whisper tiny 모델 로드
model = whisper.load_model("tiny")
model.eval()

# 오디오 인코더 변환
class WhisperEncoder(torch.nn.Module):
    def __init__(self, whisper_model):
        super().__init__()
        self.encoder = whisper_model.encoder

    def forward(self, mel):
        return self.encoder(mel)

encoder = WhisperEncoder(model)
encoder.eval()

# 트레이싱
mel_input = torch.randn(1, 80, 3000)  # 30초 오디오
traced_encoder = torch.jit.trace(encoder, mel_input)

# Core ML 변환
encoder_mlmodel = ct.convert(
    traced_encoder,
    inputs=[ct.TensorType(name="mel", shape=(1, 80, 3000))],
    minimum_deployment_target=ct.target.iOS16
)

encoder_mlmodel.save("WhisperEncoder.mlpackage")
```

성능 측정 결과입니다.

```python
WHISPER_PERFORMANCE = {
    "tiny": {
        "model_size": "39MB",
        "iphone_14_pro": {
            "inference_time": "0.8초 (30초 오디오)",
            "real_time_factor": "0.027x"  # 실시간보다 37배 빠름
        },
        "iphone_12": {
            "inference_time": "1.5초",
            "real_time_factor": "0.05x"
        }
    },
    "base": {
        "model_size": "74MB",
        "iphone_14_pro": {
            "inference_time": "1.8초",
            "real_time_factor": "0.06x"
        },
        "iphone_12": {
            "inference_time": "3.5초",
            "real_time_factor": "0.12x"
        }
    }
}
```

### 3. 텍스트 분류

DistilBERT를 모바일에 최적화해서 감정 분석에 사용했습니다.

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import coremltools as ct

# 모델 로드
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
model.eval()

# 정적 입력 크기로 래핑
class DistilBertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

wrapper = DistilBertWrapper(model)

# 트레이싱 (고정 시퀀스 길이)
MAX_SEQ_LEN = 128
dummy_input_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int32)
dummy_attention_mask = torch.ones(1, MAX_SEQ_LEN, dtype=torch.int32)

traced_model = torch.jit.trace(wrapper, (dummy_input_ids, dummy_attention_mask))

# Core ML 변환
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, MAX_SEQ_LEN), dtype=ct.int32),
        ct.TensorType(name="attention_mask", shape=(1, MAX_SEQ_LEN), dtype=ct.int32)
    ],
    minimum_deployment_target=ct.target.iOS15
)

mlmodel.save("DistilBertSentiment.mlpackage")
```

## 최적화 기법 상세

### 양자화 (Quantization)

FP32에서 INT8로 양자화하면 모델 크기가 1/4로 줄어듭니다. 정확도 손실은 대부분의 경우 1% 미만이었습니다.

```python
import torch
from torch.quantization import quantize_dynamic, get_default_qconfig
import coremltools as ct

# 방법 1: PyTorch Dynamic Quantization
model_fp32 = load_your_model()
model_int8 = quantize_dynamic(
    model_fp32,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# 방법 2: Post-Training Quantization with calibration
from torch.ao.quantization import prepare, convert, default_qconfig

model = load_your_model()
model.eval()

# 양자화 설정
model.qconfig = get_default_qconfig('fbgemm')

# 캘리브레이션 준비
model_prepared = prepare(model)

# 대표 데이터로 캘리브레이션
with torch.no_grad():
    for batch in calibration_dataloader:
        model_prepared(batch)

# 양자화 적용
model_quantized = convert(model_prepared)

# 방법 3: Core ML 양자화
mlmodel = ct.models.MLModel("model.mlpackage")

# 16비트 양자화
mlmodel_fp16 = ct.models.neural_network.quantization_utils.quantize_weights(
    mlmodel,
    nbits=16
)

# 8비트 양자화 (더 공격적)
mlmodel_int8 = ct.models.neural_network.quantization_utils.quantize_weights(
    mlmodel,
    nbits=8
)

mlmodel_int8.save("model_quantized.mlpackage")
```

양자화 결과 비교입니다.

```python
QUANTIZATION_RESULTS = {
    "MobileNetV3": {
        "fp32": {"size": "21.5MB", "accuracy": "75.2%", "latency": "12ms"},
        "fp16": {"size": "10.8MB", "accuracy": "75.1%", "latency": "8ms"},
        "int8": {"size": "5.4MB", "accuracy": "74.5%", "latency": "5ms"}
    },
    "DistilBERT": {
        "fp32": {"size": "268MB", "accuracy": "91.3%", "latency": "45ms"},
        "fp16": {"size": "134MB", "accuracy": "91.2%", "latency": "28ms"},
        "int8": {"size": "67MB", "accuracy": "90.8%", "latency": "18ms"}
    }
}
```

### Pruning (가지치기)

중요하지 않은 가중치를 제거해서 모델을 경량화합니다.

```python
import torch
import torch.nn.utils.prune as prune

def apply_structured_pruning(model, amount=0.3):
    """
    구조적 프루닝: 전체 필터/뉴런 제거
    비구조적보다 실제 속도 향상에 유리
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(
                module,
                name='weight',
                amount=amount,
                n=2,  # L2 norm 기준
                dim=0  # 출력 채널 방향
            )
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(
                module,
                name='weight',
                amount=amount
            )

    return model

def apply_iterative_pruning(model, dataloader, target_sparsity=0.5, iterations=5):
    """
    점진적 프루닝: 여러 단계에 걸쳐 조금씩 제거
    급격한 성능 저하 방지
    """
    sparsity_per_iter = 1 - (1 - target_sparsity) ** (1 / iterations)

    for i in range(iterations):
        # 프루닝 적용
        apply_structured_pruning(model, amount=sparsity_per_iter)

        # 미세 조정
        fine_tune(model, dataloader, epochs=2)

        # 프루닝 마스크 영구화
        for module in model.modules():
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')

        current_sparsity = calculate_sparsity(model)
        print(f"Iteration {i+1}: Sparsity = {current_sparsity:.2%}")

    return model

# 프루닝 전후 비교
PRUNING_RESULTS = {
    "original": {"params": "3.4M", "latency": "12ms", "accuracy": "75.2%"},
    "30%_pruned": {"params": "2.4M", "latency": "9ms", "accuracy": "74.8%"},
    "50%_pruned": {"params": "1.7M", "latency": "7ms", "accuracy": "74.1%"},
    "70%_pruned": {"params": "1.0M", "latency": "5ms", "accuracy": "72.5%"}
}
```

### Knowledge Distillation

큰 교사 모델의 지식을 작은 학생 모델로 전이합니다.

```python
import torch
import torch.nn.functional as F

class DistillationTrainer:
    def __init__(
        self,
        teacher_model,
        student_model,
        temperature=4.0,
        alpha=0.7  # soft label 가중치
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Distillation Loss = α * KL(soft_student || soft_teacher)
                          + (1-α) * CE(student, labels)
        """
        # Soft labels from teacher
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)

        # KL Divergence for soft labels
        soft_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Hard label loss
        hard_loss = F.cross_entropy(student_logits, labels)

        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

    def train_step(self, batch):
        inputs, labels = batch

        # Teacher prediction (no gradient)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)

        # Student prediction
        student_logits = self.student(inputs)

        # Calculate loss
        loss = self.distillation_loss(student_logits, teacher_logits, labels)

        return loss

# 사용 예시
teacher = models.resnet50(pretrained=True)  # 25.6M params
student = models.mobilenet_v3_small(pretrained=False)  # 2.5M params

trainer = DistillationTrainer(teacher, student, temperature=4.0)
```

## 플랫폼별 구현

### iOS (Core ML)

Core ML은 Apple의 ML 프레임워크로, Neural Engine을 활용하면 매우 빠른 추론이 가능합니다.

```swift
import CoreML

class CoreMLInference {
    private let model: MLModel
    private let asyncModel: MLModel?

    init(modelName: String) throws {
        let config = MLModelConfiguration()

        // 계산 유닛 설정
        config.computeUnits = .all  // CPU + GPU + Neural Engine

        // 비동기 예측을 위한 설정
        config.allowLowPrecisionAccumulationOnGPU = true

        // 모델 로드
        guard let modelURL = Bundle.main.url(
            forResource: modelName,
            withExtension: "mlpackage"
        ) else {
            throw ModelError.modelNotFound
        }

        model = try MLModel(contentsOf: modelURL, configuration: config)

        // iOS 16+ 비동기 모델
        if #available(iOS 16.0, *) {
            asyncModel = model
        } else {
            asyncModel = nil
        }
    }

    // 동기 추론
    func predict(input: MLFeatureProvider) throws -> MLFeatureProvider {
        return try model.prediction(from: input)
    }

    // 비동기 추론 (iOS 16+)
    @available(iOS 16.0, *)
    func predictAsync(input: MLFeatureProvider) async throws -> MLFeatureProvider {
        return try await asyncModel!.prediction(from: input)
    }

    // 배치 추론
    func predictBatch(inputs: [MLFeatureProvider]) throws -> [MLFeatureProvider] {
        let batchProvider = MLArrayBatchProvider(array: inputs)
        let results = try model.predictions(fromBatch: batchProvider)

        var outputs: [MLFeatureProvider] = []
        for i in 0..<results.count {
            outputs.append(results.features(at: i))
        }
        return outputs
    }
}

// 성능 측정
class PerformanceProfiler {
    static func measureInference(
        model: MLModel,
        input: MLFeatureProvider,
        iterations: Int = 100
    ) -> (mean: Double, std: Double) {
        var times: [Double] = []

        // 워밍업
        for _ in 0..<10 {
            _ = try? model.prediction(from: input)
        }

        // 측정
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try? model.prediction(from: input)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            times.append(elapsed * 1000)  // ms로 변환
        }

        let mean = times.reduce(0, +) / Double(times.count)
        let variance = times.map { pow($0 - mean, 2) }.reduce(0, +) / Double(times.count)
        let std = sqrt(variance)

        return (mean, std)
    }
}
```

### Android (TensorFlow Lite)

TensorFlow Lite는 Android에서 가장 널리 사용되는 ML 프레임워크입니다.

```kotlin
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TFLiteInference(context: Context, modelPath: String) {
    private val interpreter: Interpreter

    init {
        // 모델 로드
        val modelBuffer = loadModelFile(context, modelPath)

        // 인터프리터 옵션 설정
        val options = Interpreter.Options().apply {
            // GPU 가속 (지원되는 기기에서)
            try {
                val gpuDelegate = GpuDelegate(
                    GpuDelegate.Options().apply {
                        setPrecisionLossAllowed(true)  // FP16 허용
                        setInferencePreference(
                            GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED
                        )
                    }
                )
                addDelegate(gpuDelegate)
            } catch (e: Exception) {
                // GPU 미지원 기기 - CPU fallback
            }

            // NNAPI 가속 (Android 8.1+)
            try {
                val nnApiDelegate = NnApiDelegate(
                    NnApiDelegate.Options().apply {
                        setAllowFp16(true)
                        setUseNnapiCpu(false)
                    }
                )
                addDelegate(nnApiDelegate)
            } catch (e: Exception) {
                // NNAPI 미지원
            }

            setNumThreads(4)
        }

        interpreter = Interpreter(modelBuffer, options)
    }

    fun predict(input: FloatArray): FloatArray {
        val inputBuffer = ByteBuffer.allocateDirect(input.size * 4).apply {
            order(ByteOrder.nativeOrder())
            input.forEach { putFloat(it) }
            rewind()
        }

        val outputShape = interpreter.getOutputTensor(0).shape()
        val outputSize = outputShape.reduce { acc, i -> acc * i }
        val outputBuffer = ByteBuffer.allocateDirect(outputSize * 4).apply {
            order(ByteOrder.nativeOrder())
        }

        interpreter.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val output = FloatArray(outputSize)
        outputBuffer.asFloatBuffer().get(output)

        return output
    }

    // 이미지 분류용 편의 메서드
    fun classifyImage(bitmap: Bitmap): List<Pair<Int, Float>> {
        val inputArray = preprocessImage(bitmap)
        val output = predict(inputArray)

        return output.mapIndexed { index, confidence ->
            index to confidence
        }.sortedByDescending { it.second }.take(5)
    }

    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val pixels = IntArray(224 * 224)
        resized.getPixels(pixels, 0, 224, 0, 0, 224, 224)

        val input = FloatArray(224 * 224 * 3)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            input[i * 3] = ((pixel shr 16 and 0xFF) / 255.0f - 0.485f) / 0.229f
            input[i * 3 + 1] = ((pixel shr 8 and 0xFF) / 255.0f - 0.456f) / 0.224f
            input[i * 3 + 2] = ((pixel and 0xFF) / 255.0f - 0.406f) / 0.225f
        }

        return input
    }

    private fun loadModelFile(context: Context, path: String): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(path)
        val inputStream = assetFileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(
            java.nio.channels.FileChannel.MapMode.READ_ONLY,
            startOffset,
            declaredLength
        )
    }

    fun close() {
        interpreter.close()
    }
}
```

## 성능 벤치마크

실제 디바이스에서 측정한 결과입니다.

```python
BENCHMARK_RESULTS = {
    "MobileNetV3-Small (Image Classification)": {
        "iPhone_14_Pro": {"latency": "3.2ms", "throughput": "312 FPS"},
        "iPhone_12": {"latency": "5.8ms", "throughput": "172 FPS"},
        "Pixel_7": {"latency": "4.5ms", "throughput": "222 FPS"},
        "Galaxy_S23": {"latency": "4.1ms", "throughput": "244 FPS"},
        "Pixel_4a": {"latency": "12.3ms", "throughput": "81 FPS"}
    },
    "Whisper-Tiny (30s Audio)": {
        "iPhone_14_Pro": {"latency": "0.8s", "RTF": "0.027"},
        "iPhone_12": {"latency": "1.5s", "RTF": "0.05"},
        "Pixel_7": {"latency": "1.2s", "RTF": "0.04"},
        "Galaxy_S23": {"latency": "1.1s", "RTF": "0.037"}
    },
    "DistilBERT (128 tokens)": {
        "iPhone_14_Pro": {"latency": "15ms"},
        "iPhone_12": {"latency": "28ms"},
        "Pixel_7": {"latency": "22ms"},
        "Galaxy_S23": {"latency": "19ms"}
    }
}
```

## 배터리 및 발열 고려사항

모바일 AI에서 간과하기 쉬운 부분이 배터리 소모와 발열입니다.

```python
POWER_CONSUMPTION = {
    "continuous_inference": {
        "warning": "지속적인 추론은 배터리를 빠르게 소모",
        "image_classification": {
            "30fps_continuous": "시간당 약 15-20% 배터리 소모",
            "recommendation": "필요할 때만 추론, 프레임 스킵 고려"
        },
        "audio_processing": {
            "realtime_transcription": "시간당 약 10-15% 배터리 소모",
            "recommendation": "버퍼링 후 배치 처리 고려"
        }
    },
    "thermal_throttling": {
        "issue": "지속적 추론 시 발열로 인한 성능 저하",
        "mitigation": [
            "추론 간격 두기 (예: 100ms마다)",
            "Neural Engine 우선 사용 (GPU보다 발열 적음)",
            "배치 크기 조절"
        ]
    }
}

# 배터리 효율적인 추론 패턴
class BatteryEfficientInference:
    def __init__(self, model, min_interval_ms=100):
        self.model = model
        self.min_interval = min_interval_ms / 1000
        self.last_inference_time = 0

    def should_run_inference(self):
        """필요할 때만 추론 실행"""
        current_time = time.time()
        if current_time - self.last_inference_time >= self.min_interval:
            self.last_inference_time = current_time
            return True
        return False

    def run_with_throttling(self, input_data):
        if not self.should_run_inference():
            return self.last_result  # 캐시된 결과 반환

        self.last_result = self.model.predict(input_data)
        return self.last_result
```

## 한계와 현실

2023년 기준, 아직 대형 모델은 모바일에서 실용적이지 않습니다.

```python
MOBILE_AI_LIMITATIONS = {
    "model_size": {
        "practical_limit": "~500MB",
        "reason": "앱 다운로드 크기, 메모리 제약",
        "llm_7b": "약 4GB (4-bit 양자화 후에도)",
        "feasibility": "아직 어려움"
    },
    "memory": {
        "iphone_14_pro": "6GB RAM",
        "typical_android": "4-8GB RAM",
        "available_for_ml": "1-2GB (다른 앱, OS 고려)",
        "llm_requirement": "4GB+ (7B 모델)"
    },
    "battery": {
        "issue": "지속 사용 시 빠른 배터리 소모",
        "user_experience": "영향 큼"
    }
}
```

## 2024년 전망

하드웨어가 계속 발전하면서 더 큰 모델도 모바일에서 돌릴 수 있게 될 것입니다.

```python
FUTURE_OUTLOOK = {
    "hardware_improvements": {
        "neural_engine": "매년 2-3배 성능 향상",
        "memory": "8GB+ RAM 일반화",
        "npu": "전용 AI 칩 탑재 확대"
    },
    "software_optimizations": {
        "quantization": "2-bit, 1-bit 양자화 연구",
        "speculative_decoding": "작은 모델로 큰 모델 가속",
        "continuous_batching": "효율적인 배치 처리"
    },
    "expected_capabilities": {
        "2024": "3B 파라미터 LLM 모바일 실행",
        "2025": "7B+ 모델 실용화 예상"
    }
}
```

llama.cpp로 7B 모델을 맥북에서 돌리는 것은 이미 가능하고, 일부 사용자는 iPhone에서도 실행하고 있습니다. 아직 실용적인 속도는 아니지만, 방향성은 명확합니다.

## 참고 자료

- [Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [ONNX Runtime Mobile](https://onnxruntime.ai/docs/tutorials/mobile/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM의 모바일/엣지 실행
- [MLC LLM](https://mlc.ai/mlc-llm/) - 범용 LLM 배포 솔루션

