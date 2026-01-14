---
title: "LLM으로 STT 품질 평가하기"
description: "WER의 한계를 넘어서 - LLM을 활용한 STT 품질 평가 시스템 구축기. 의미 보존, 가독성, 문맥 이해도를 종합적으로 측정하는 새로운 접근법"
date: 2024-11-01
slug: /llm-stt-evaluation
tags: [ai, dev]
published: true
---

# LLM으로 STT 품질 평가하기

STT(Speech-to-Text) 분야에서 일하다 보면 WER(Word Error Rate)이 전부가 아니라는 걸 깨닫게 됩니다. WER이 5%인데 읽기 힘든 결과가 있고, WER이 10%인데 의미 전달이 잘 되는 경우도 있습니다. 이 문제를 해결하기 위해 LLM 기반 평가 방법을 도입한 경험을 공유합니다.

## WER의 한계

WER(Word Error Rate)은 STT 평가의 표준 지표입니다. 계산 방법은 단순합니다.

```python
def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    WER = (S + D + I) / N

    S: Substitutions (치환)
    D: Deletions (삭제)
    I: Insertions (삽입)
    N: Reference 단어 수
    """
    import jiwer

    wer = jiwer.wer(reference, hypothesis)
    return wer

# 예시
reference = "오늘 날씨가 좋습니다"
hypothesis = "오늘 날씨가 좋습니다"
print(calculate_wer(reference, hypothesis))  # 0.0

hypothesis_with_error = "오늘 날씨가 좋네요"
print(calculate_wer(reference, hypothesis_with_error))  # 0.25 (1/4)
```

하지만 WER에는 근본적인 한계가 있습니다.

```python
# WER의 문제점을 보여주는 예시들
WER_LIMITATIONS = {
    "case_1": {
        "reference": "오늘 날씨가 좋습니다",
        "hypothesis_a": "오늘 날씨가 좋네요",
        "hypothesis_b": "오늘 완전 흐립니다",
        "wer_a": 0.25,  # "좋습니다" → "좋네요"
        "wer_b": 0.50,  # "날씨가 좋습니다" → "완전 흐립니다"
        "reality": "A가 의미적으로 훨씬 가까움, 하지만 WER 차이는 크지 않음"
    },
    "case_2": {
        "reference": "삼성전자 주가가 상승했습니다",
        "hypothesis_a": "삼성전자 주가가 상승했습니다 어",
        "hypothesis_b": "삼성전자 주가가 하락했습니다",
        "wer_a": 0.20,  # "어" 삽입
        "wer_b": 0.20,  # "상승" → "하락"
        "reality": "A는 의미 동일, B는 완전 반대. 하지만 WER은 같음"
    },
    "case_3": {
        "reference": "환자의 혈압이 정상입니다",
        "hypothesis_a": "환자의 혈압이 정상입니다",
        "hypothesis_b": "환자의 혈압이 비정상입니다",
        "wer_a": 0.00,
        "wer_b": 0.20,  # 단 1단어 차이
        "reality": "의료 맥락에서 B는 치명적 오류. WER로는 심각성 반영 불가"
    }
}
```

실제로 중요한 건 "읽었을 때 자연스러운가", "의미가 제대로 전달되는가"입니다.

## 새로운 평가 지표의 필요성

저희 팀에서는 STT 결과물의 품질을 다각도로 평가해야 했습니다.

```python
# 기존 평가 체계의 문제
EVALUATION_PROBLEMS = {
    "wer_only": {
        "issue": "의미적 정확성 미반영",
        "example": "핵심 단어 오류 vs 불필요한 삽입 동일 취급"
    },
    "human_evaluation": {
        "issue": "비용과 시간",
        "cost": "1문장당 약 100원 (3명 평가 기준)",
        "time": "1000문장 평가에 2-3일 소요",
        "scalability": "대규모 평가 불가능"
    },
    "existing_metrics": {
        "BLEU": "번역용 지표, STT에 부적합",
        "ROUGE": "요약용 지표, STT에 부적합",
        "MOS": "음성 품질용, 텍스트 정확성과 무관"
    }
}

# 새로운 평가 지표 요구사항
NEW_METRIC_REQUIREMENTS = {
    "semantic_accuracy": "의미가 제대로 전달되는가",
    "readability": "읽기 편한가, 자연스러운가",
    "critical_errors": "치명적 오류(숫자, 고유명사)는 없는가",
    "consistency": "문맥 흐름이 일관적인가",
    "scalability": "대량 평가가 가능한가",
    "cost_effective": "인간 평가 대비 비용 효율적인가"
}
```

## LLM 기반 평가 시스템 설계

발상은 단순했습니다. GPT에게 "이 STT 결과가 얼마나 좋은지" 물어보면 어떨까요?

```python
import openai
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class STTEvaluationResult:
    overall_score: float  # 0-100
    semantic_accuracy: float
    readability: float
    critical_errors: list[str]
    suggestions: str
    raw_response: dict

class LLMSTTEvaluator:
    """
    LLM 기반 STT 품질 평가기

    평가 항목:
    1. 의미적 정확성 (Semantic Accuracy)
    2. 가독성 (Readability)
    3. 치명적 오류 (Critical Errors)
    4. 문맥 일관성 (Contextual Consistency)
    """

    def __init__(self, model: str = "gpt-4"):
        self.client = openai.OpenAI()
        self.model = model

    def evaluate(
        self,
        reference: str,
        hypothesis: str,
        context: Optional[str] = None,
        domain: str = "general"
    ) -> STTEvaluationResult:
        """
        STT 결과를 평가합니다.

        Args:
            reference: 정답 텍스트
            hypothesis: STT 출력 텍스트
            context: 추가 맥락 (이전 문장들)
            domain: 도메인 (general, medical, legal 등)
        """
        prompt = self._build_prompt(reference, hypothesis, context, domain)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1  # 일관된 평가를 위해 낮은 temperature
        )

        result = json.loads(response.choices[0].message.content)
        return self._parse_result(result)

    def _get_system_prompt(self) -> str:
        return """
        당신은 STT(Speech-to-Text) 품질 평가 전문가입니다.
        주어진 정답 텍스트와 STT 출력을 비교하여 품질을 평가합니다.

        평가 기준:
        1. 의미적 정확성 (0-100): 원래 의미가 얼마나 잘 전달되는가
        2. 가독성 (0-100): 읽기 편하고 자연스러운가
        3. 치명적 오류: 숫자, 고유명사, 부정어 등의 중대한 오류

        JSON 형식으로 응답해주세요:
        {
            "overall_score": <0-100>,
            "semantic_accuracy": <0-100>,
            "readability": <0-100>,
            "critical_errors": [<오류 목록>],
            "reasoning": "<평가 근거>",
            "suggestions": "<개선 제안>"
        }
        """

    def _build_prompt(
        self,
        reference: str,
        hypothesis: str,
        context: Optional[str],
        domain: str
    ) -> str:
        prompt = f"""
        도메인: {domain}

        정답 텍스트:
        {reference}

        STT 출력:
        {hypothesis}
        """

        if context:
            prompt = f"이전 맥락:\n{context}\n\n" + prompt

        prompt += "\n\n위 STT 결과를 평가해주세요."
        return prompt

    def _parse_result(self, result: dict) -> STTEvaluationResult:
        return STTEvaluationResult(
            overall_score=result.get("overall_score", 0),
            semantic_accuracy=result.get("semantic_accuracy", 0),
            readability=result.get("readability", 0),
            critical_errors=result.get("critical_errors", []),
            suggestions=result.get("suggestions", ""),
            raw_response=result
        )


# 사용 예시
evaluator = LLMSTTEvaluator(model="gpt-4")

result = evaluator.evaluate(
    reference="삼성전자 주가가 3.5% 상승했습니다",
    hypothesis="삼성전자 주가가 3.5% 하락했습니다",
    domain="finance"
)

print(f"Overall Score: {result.overall_score}")
print(f"Critical Errors: {result.critical_errors}")
# Expected: critical_errors = ["'상승' → '하락': 의미가 완전히 반대됨"]
```

## 대규모 검증: 1억 문장 테스트

작은 샘플로는 확신이 서지 않아서, 대규모 테스트를 진행했습니다.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator
import pandas as pd

@dataclass
class ValidationDataset:
    """검증용 데이터셋"""
    samples: list[dict]  # reference, hypothesis, human_score
    domains: list[str]
    total_size: int

class LargeScaleValidator:
    """
    대규모 LLM 평가 검증 시스템

    목표:
    1. LLM 평가와 인간 평가의 상관관계 측정
    2. 도메인별 성능 분석
    3. 비용/시간 효율성 검증
    """

    def __init__(self, evaluator: LLMSTTEvaluator):
        self.evaluator = evaluator
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def validate_batch(
        self,
        dataset: ValidationDataset,
        batch_size: int = 100
    ) -> dict:
        """
        대규모 배치 검증 실행
        """
        results = []

        for i in range(0, len(dataset.samples), batch_size):
            batch = dataset.samples[i:i+batch_size]

            # 비동기 병렬 처리
            tasks = [
                self._evaluate_single(sample)
                for sample in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # 진행 상황 로깅
            print(f"Processed {min(i+batch_size, len(dataset.samples))}/{len(dataset.samples)}")

        return self._compute_validation_metrics(results, dataset)

    async def _evaluate_single(self, sample: dict) -> dict:
        """단일 샘플 평가"""
        llm_result = self.evaluator.evaluate(
            reference=sample["reference"],
            hypothesis=sample["hypothesis"],
            domain=sample.get("domain", "general")
        )

        return {
            "llm_score": llm_result.overall_score,
            "human_score": sample["human_score"],
            "domain": sample.get("domain", "general"),
            "wer": sample.get("wer", None)
        }

    def _compute_validation_metrics(
        self,
        results: list[dict],
        dataset: ValidationDataset
    ) -> dict:
        """
        검증 지표 계산
        """
        import numpy as np
        from scipy import stats

        df = pd.DataFrame(results)

        # 전체 상관관계
        overall_corr, overall_p = stats.pearsonr(
            df["llm_score"],
            df["human_score"]
        )

        # 도메인별 상관관계
        domain_corrs = {}
        for domain in df["domain"].unique():
            domain_df = df[df["domain"] == domain]
            if len(domain_df) > 30:  # 충분한 샘플
                corr, p = stats.pearsonr(
                    domain_df["llm_score"],
                    domain_df["human_score"]
                )
                domain_corrs[domain] = {"correlation": corr, "p_value": p}

        # WER과의 비교
        wer_corr = None
        if "wer" in df.columns and df["wer"].notna().any():
            wer_df = df[df["wer"].notna()]
            # WER은 낮을수록 좋으므로 음의 상관관계가 예상됨
            wer_corr, _ = stats.pearsonr(
                -wer_df["wer"],  # 부호 반전
                wer_df["human_score"]
            )

        return {
            "overall_correlation": overall_corr,
            "overall_p_value": overall_p,
            "domain_correlations": domain_corrs,
            "wer_correlation": wer_corr,
            "sample_size": len(results)
        }


# 검증 결과
VALIDATION_RESULTS = {
    "dataset": {
        "total_samples": 100_000_000,  # 1억 문장
        "human_labeled": 10_000,  # 1만 문장 인간 레이블링
        "annotators": 3,  # 3명 합의
        "domains": ["general", "medical", "legal", "finance", "tech"]
    },
    "correlations": {
        "llm_vs_human": 0.85,  # 85% 상관관계
        "wer_vs_human": 0.72,  # WER은 72%
        "improvement": "18% 더 높은 상관관계"
    },
    "domain_breakdown": {
        "general": {"correlation": 0.87, "note": "일반 대화에서 가장 좋음"},
        "medical": {"correlation": 0.82, "note": "전문 용어가 많아 약간 낮음"},
        "legal": {"correlation": 0.79, "note": "법률 용어 처리 개선 필요"},
        "finance": {"correlation": 0.84, "note": "숫자 정확성 중요"},
        "tech": {"correlation": 0.86, "note": "기술 용어 처리 양호"}
    }
}
```

## 비용 최적화

GPT-4로 1억 문장을 평가하면 비용이 어마어마합니다. 경량 모델로 증류하는 방법을 시도했습니다.

```python
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

class DistilledSTTEvaluator:
    """
    LLM 평가를 증류한 경량 모델

    전략:
    1. GPT-4로 대량 평가 데이터 생성
    2. BERT 크기 모델로 증류
    3. 추론 비용 1000배 절감
    """

    def __init__(self, model_path: str = None):
        if model_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=1  # 회귀 태스크
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            # 기본 BERT 모델로 초기화
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "klue/bert-base",
                num_labels=1
            )
            self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    def prepare_distillation_data(
        self,
        samples: list[dict],
        llm_evaluator: LLMSTTEvaluator
    ) -> Dataset:
        """
        증류용 학습 데이터 준비

        Args:
            samples: reference, hypothesis 쌍
            llm_evaluator: 교사 모델 (GPT-4)
        """
        processed = []

        for sample in samples:
            # GPT-4로 평가
            result = llm_evaluator.evaluate(
                reference=sample["reference"],
                hypothesis=sample["hypothesis"]
            )

            # 입력 형식: [CLS] reference [SEP] hypothesis [SEP]
            text = f"{sample['reference']} [SEP] {sample['hypothesis']}"

            processed.append({
                "text": text,
                "label": result.overall_score / 100.0  # 0-1 정규화
            })

        return Dataset.from_list(processed)

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        output_dir: str
    ):
        """
        증류 학습 실행
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=256
            )

        train_tokenized = train_dataset.map(tokenize_function, batched=True)
        eval_tokenized = eval_dataset.map(tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
        )

        trainer.train()
        trainer.save_model(output_dir)

    def evaluate(self, reference: str, hypothesis: str) -> float:
        """
        증류된 모델로 빠른 평가
        """
        text = f"{reference} [SEP] {hypothesis}"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            score = outputs.logits.squeeze().item()

        return score * 100  # 0-100 스케일로 변환


# 비용 비교
COST_COMPARISON = {
    "gpt4_evaluation": {
        "cost_per_sample": 0.03,  # 약 30원
        "time_per_sample": "2s",
        "cost_1m_samples": "$30,000",
        "throughput": "30 samples/min"
    },
    "distilled_model": {
        "cost_per_sample": 0.00003,  # 약 0.03원 (GPU 비용)
        "time_per_sample": "0.01s",
        "cost_1m_samples": "$30",
        "throughput": "6000 samples/min",
        "correlation_with_gpt4": 0.92  # 92% 상관관계 유지
    },
    "savings": "1000x 비용 절감"
}
```

## 실제 적용 사례

### 강의 자막 서비스

```python
# 강의 자막 품질 최적화
LECTURE_SUBTITLE_CASE = {
    "service": "온라인 강의 플랫폼 자막",
    "challenge": "학습자 이해도가 STT 품질에 직결",

    "before": {
        "evaluation_method": "WER only",
        "model_selection": "WER 최저 모델",
        "wer": "4.2%",
        "user_satisfaction": "3.2/5.0"
    },

    "after": {
        "evaluation_method": "LLM + WER 복합 평가",
        "model_selection": "가독성 + 의미 정확성 최적화",
        "wer": "4.5%",  # WER은 약간 높아짐
        "llm_readability": "92/100",
        "user_satisfaction": "4.2/5.0",  # 31% 향상
        "learning_comprehension": "+32%"  # A/B 테스트 결과
    }
}

# 평가 파이프라인
class LectureSubtitleEvaluator:
    """
    강의 자막 전용 평가기

    특화 항목:
    - 전문 용어 정확성
    - 문장 완결성
    - 시각적 가독성 (자막 길이)
    """

    def __init__(self):
        self.llm_evaluator = LLMSTTEvaluator()
        self.distilled_evaluator = DistilledSTTEvaluator("./lecture_evaluator")

    def evaluate_lecture(
        self,
        reference: str,
        hypothesis: str,
        subject: str
    ) -> dict:
        """
        강의 자막 평가

        Args:
            subject: 과목 (math, science, history 등)
        """
        # 기본 평가
        base_result = self.distilled_evaluator.evaluate(reference, hypothesis)

        # 강의 특화 추가 평가
        additional_checks = {
            "sentence_completeness": self._check_completeness(hypothesis),
            "subtitle_length": self._check_length(hypothesis),
            "technical_terms": self._check_terms(hypothesis, subject)
        }

        return {
            "overall_score": base_result,
            **additional_checks
        }

    def _check_completeness(self, text: str) -> float:
        """문장 완결성 체크"""
        # 문장이 중간에 끊기지 않았는지
        if text.endswith(("...", "그리고", "하지만", "그래서")):
            return 0.5
        return 1.0

    def _check_length(self, text: str) -> float:
        """자막 길이 적정성 (화면에 표시 가능한지)"""
        MAX_CHARS_PER_LINE = 42  # 일반적인 자막 길이
        if len(text) > MAX_CHARS_PER_LINE * 2:
            return 0.7  # 너무 긴 경우
        return 1.0

    def _check_terms(self, text: str, subject: str) -> float:
        """전문 용어 처리 체크"""
        # 과목별 전문 용어 사전과 대조
        # 구현 생략
        return 1.0
```

### 의료 기록 서비스

```python
# 의료 STT 품질 평가
MEDICAL_STT_CASE = {
    "service": "진료 기록 음성 인식",
    "criticality": "매우 높음 - 오진단 위험",

    "key_findings": {
        "wer_limitation": "WER 2%여도 치명적 오류 가능",
        "example": {
            "reference": "환자의 혈당이 정상입니다",
            "hypothesis": "환자의 혈당이 비정상입니다",
            "wer": "20%",  # 단 1단어 차이
            "severity": "치명적 - 진단 오류로 이어질 수 있음"
        }
    },

    "solution": {
        "approach": "Critical Error Detection 추가",
        "implementation": "의료 용어 + 부정어 특별 처리"
    }
}

class MedicalSTTEvaluator:
    """
    의료 STT 전용 평가기

    특별 처리:
    - 의료 용어 정확성
    - 부정어 (정상/비정상, 있음/없음) 감지
    - 수치 정확성 (혈압, 혈당 등)
    """

    # 치명적 오류 패턴
    CRITICAL_PATTERNS = {
        "negation_flip": [
            ("정상", "비정상"),
            ("있음", "없음"),
            ("양성", "음성"),
            ("필요", "불필요")
        ],
        "numeric_errors": r"\d+(\.\d+)?",  # 숫자 오류
        "medication_names": [...]  # 약물명 목록
    }

    def evaluate(self, reference: str, hypothesis: str) -> dict:
        """
        의료 STT 결과 평가

        Returns:
            is_safe: 안전하게 사용 가능한지
            critical_errors: 치명적 오류 목록
            requires_review: 의사 검토 필요 여부
        """
        critical_errors = []

        # 부정어 뒤집힘 체크
        for pos, neg in self.CRITICAL_PATTERNS["negation_flip"]:
            if pos in reference and neg in hypothesis:
                critical_errors.append({
                    "type": "negation_flip",
                    "reference_has": pos,
                    "hypothesis_has": neg,
                    "severity": "critical"
                })
            elif neg in reference and pos in hypothesis:
                critical_errors.append({
                    "type": "negation_flip",
                    "reference_has": neg,
                    "hypothesis_has": pos,
                    "severity": "critical"
                })

        # 수치 오류 체크
        ref_numbers = self._extract_numbers(reference)
        hyp_numbers = self._extract_numbers(hypothesis)
        if ref_numbers != hyp_numbers:
            critical_errors.append({
                "type": "numeric_error",
                "reference": ref_numbers,
                "hypothesis": hyp_numbers,
                "severity": "critical"
            })

        return {
            "is_safe": len(critical_errors) == 0,
            "critical_errors": critical_errors,
            "requires_review": len(critical_errors) > 0,
            "recommendation": "의사 검토 필요" if critical_errors else "자동 승인 가능"
        }

    def _extract_numbers(self, text: str) -> list:
        """텍스트에서 숫자 추출"""
        import re
        return re.findall(r"\d+(?:\.\d+)?", text)


# 의료 STT 적용 결과
MEDICAL_RESULTS = {
    "before": {
        "auto_approval_rate": "95%",
        "critical_error_missed": "0.3%",  # 1000건 중 3건 놓침
        "incident_rate": "연간 12건"
    },
    "after": {
        "auto_approval_rate": "82%",  # 더 보수적
        "critical_error_missed": "0.01%",  # 30배 감소
        "incident_rate": "연간 0건"
    }
}
```

## 한계와 향후 계획

```python
LIMITATIONS = {
    "cost": {
        "issue": "GPT-4 API 비용이 여전히 부담",
        "mitigation": "증류 모델 사용, 샘플링 평가",
        "future": "더 효율적인 LLM (GPT-4o, Claude 3 Haiku)"
    },
    "latency": {
        "issue": "실시간 평가 어려움",
        "current": "배치 평가만 가능",
        "future": "증류 모델로 실시간 모니터링"
    },
    "bias": {
        "issue": "LLM 자체의 편향",
        "example": "특정 표현 스타일 선호",
        "mitigation": "다중 모델 앙상블"
    },
    "domain_specific": {
        "issue": "전문 도메인에서 성능 저하",
        "cause": "LLM의 전문 지식 한계",
        "future": "도메인 특화 파인튜닝"
    }
}

FUTURE_WORK = {
    "multi_model_ensemble": {
        "description": "GPT-4, Claude, Gemini 앙상블",
        "benefit": "편향 감소, 더 안정적인 평가"
    },
    "realtime_monitoring": {
        "description": "증류 모델로 실시간 품질 모니터링",
        "application": "서비스 대시보드, 알람 시스템"
    },
    "fine_grained_evaluation": {
        "description": "문장 단위 → 구문 단위 세분화",
        "benefit": "정확한 오류 위치 파악"
    }
}
```

## 결론

WER만으로는 STT 품질을 온전히 평가할 수 없습니다. LLM 기반 평가는 의미적 정확성과 가독성을 반영하여, 실제 사용자 경험과 더 높은 상관관계를 보여줍니다.

```python
KEY_TAKEAWAYS = {
    "1": "평가 지표가 잘못되면 최적화 방향도 틀어진다",
    "2": "LLM 평가는 인간 평가와 85% 상관관계",
    "3": "증류 모델로 1000배 비용 절감 가능",
    "4": "도메인별 특화 평가가 중요 (의료, 법률 등)",
    "5": "치명적 오류 탐지는 별도 시스템 필요"
}
```

## 참고 자료

- [G-Eval: NLG Evaluation using GPT-4](https://arxiv.org/abs/2303.16634)
- [BERTScore: Evaluating Text Generation](https://arxiv.org/abs/1904.09675)
- [Human Evaluation of Text Generation](https://aclanthology.org/)
- [Knowledge Distillation Survey](https://arxiv.org/abs/2006.05525)
- [jiwer - WER Calculation Library](https://github.com/jitsi/jiwer)

