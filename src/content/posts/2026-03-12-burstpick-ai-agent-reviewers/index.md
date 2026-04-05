---
title: "BurstPick: 4명의 AI 전문가가 리뷰하는 앱 개발 프로세스"
description: "사진 큐레이션 앱 BurstPick의 개발에서 AI 에이전트 리뷰어를 활용한 체계적인 코드 품질 관리 경험을 공유합니다."
date: 2026-03-12
permalink: /burstpick-ai-agent-reviewers
tags: [AI, CodeReview, Swift, MacOS, Windows, ML, DevProcess]
published: true
---

# BurstPick: 4명의 AI 전문가가 리뷰하는 앱 개발 프로세스

## 들어가며

[BurstPick](https://burstpick.app)은 사진작가를 위한 버스트 사진 큐레이션 앱입니다. 세션당 3,000장 이상의 연사 프레임에서 최고의 컷을 자동으로 골라내는 도구죠. 22개의 온디바이스 ML 모델, 39개 언어 지원, macOS와 Windows 동시 개발 — 혼자서 이 규모의 프로젝트를 관리하려면 코드 리뷰가 생명입니다.

문제는, 1인 개발자에게 리뷰어가 없다는 점입니다. 아니, 없었습니다. 지금은 4명의 AI 전문가가 매 리뷰 라운드마다 동시에 코드를 검토합니다. 45번의 리뷰 라운드를 거치며 다듬어진 이 프로세스를 공유합니다.

## 4명의 전문 리뷰어

BurstPick의 리뷰 시스템에는 4명의 전문가가 있습니다. 각자 완전히 다른 렌즈로 코드를 봅니다.

### 1. Pro Photographer Reviewer — 현업 관점

"내가 내일 당장 이걸 내 비즈니스에서 쓸 수 있나?"

이 리뷰어는 기능 완성도, 워크플로우 통합, 데이터 안전성을 봅니다. 코드의 품질보다 사용자의 신뢰를 우선시해요.

R45 점수: **8.1/10** — "디자인 토큰 정리는 폴리시 작업이지, 기능 진전은 아니다"

### 2. Systems Performance Engineer — 성능 관점

"이거 10K장에서도 스케일하나? 핫 패스가 어딘가?"

CPU 활용, 메모리 프로필, GPU 가속, 동시성 패턴을 검사합니다. 벤치마크 수치로 이야기하는 리뷰어.

R45 점수: **9.2/10** — "디자인 토큰 마이그레이션에서 성능 임팩트 제로. 리그레션 없음"

### 3. Product Marketing Reviewer — 시장 관점

"사진작가들이 이게 Photo Mechanic과 뭐가 다른지 이해하겠나?"

가치 제안의 명확성, 경쟁 차별화, GTM 준비 상태를 평가합니다.

R45 점수: **9.3/10**

### 4. UI/UX Design Reviewer — 인터페이스 관점

"디자인 시스템이 일관적인가? 접근성 패턴이 적용되었나?"

디자인 토큰 일관성, 접근성 어노테이션, 시각적 위계를 검사합니다.

R45 점수: **9.5/10** — "UI-11, UI-12 수정 확인 완료"

## 리뷰 라운드 시스템

45번의 라운드를 거치면서 정립된 프로세스입니다.

### 라운드 구조

```
R44: 변경 사항 리뷰 요청
  ↓
4명의 리뷰어가 동시에 리뷰 (각자 다른 LLM)
  ↓
각 리뷰어가 리포트 생성
  ↓
통합 리포트로 머지
  ↓
이슈 수정 (CRITICAL → HIGH → MEDIUM 순서)
  ↓
R45: 수정 사항 확인 리뷰
```

각 리뷰어는 자기 분야만 봅니다. 보안 리뷰어가 UI 디자인을 지적하거나, 마케팅 리뷰어가 코드 품질을 논하지 않습니다. 이 경계가 리뷰 품질을 높이는 핵심입니다.

### Ground Truth 문서

리뷰가 반복되면서 한 가지 문제가 생겼습니다: 같은 이슈가 계속 재등장하는 것. 이미 해결된 항목을 리뷰어가 다시 지적하거나, 설계상 의도된 패턴을 문제로 판단하거나.

이를 해결하기 위해 `REVIEW-GROUND-TRUTH.md`를 만들었습니다:

```markdown
## Resolved Items
| ID | Issue | Status | Round Fixed |
|----|-------|--------|-------------|
| SEC-1 | API key in source | Fixed | R12 |
| CQ-3 | AppState 5,800 lines | Accepted (MEDIUM debt) | R18 |
| PERF-7 | Vision framework deadlock | Fixed (4 concurrent limit) | R23 |

## Accepted Patterns
- ~190 `try?` expressions — intentional for non-critical image operations
- ~12 `@unchecked Sendable` — reviewed, minimal actor isolation bypass
- ~140 `.shared` singleton accesses — standard service layer pattern
```

이 문서가 있으면 리뷰어들이 이미 해결된 문제를 재론하지 않습니다. 점수도 단조 비감소(monotonically non-decreasing)여야 한다는 규칙을 적용했어요. 수정이 적용되었으면 점수가 내려가서는 안 됩니다.

### 심각도 체계

리뷰어 간 일관성을 위한 엄격한 정의:

| 심각도 | 정의 | 예시 |
|--------|------|------|
| CRITICAL | 데이터 손실, 보안 익스플로잇, 크래시, 핵심 기능 고장 | RAW 파일 손상, XMP 덮어쓰기 |
| HIGH | 드문 크래시, 2배 이상 성능 저하, 부분 기능 실패 | 10K장에서 OOM, 클러스터링 실패 |
| MEDIUM | 기술 부채, 성능 이슈, 비핵심 기능 누락 | 5,800줄 AppState, 캐시 미스 |
| LOW | 스타일, 최적화, 문서 | 변수명, 주석 누락 |

## 달러 비용으로 이야기하기

가장 효과적이었던 도입은 **달러 비용(Dollar Cost)**입니다.

리뷰어가 이슈를 보고할 때 "이걸 안 고치면 얼마의 비용이 발생하는가"를 추정합니다:

| 이슈 | 상태 | 달러 비용 | 설명 |
|------|------|-----------|------|
| CG-1: Lightroom 계층 키워드 미지원 | Open | $200-500/건 | 키워드 수동 재작업 비용 |
| CG-3: 그리드에 임베디드 JPEG 미사용 | Open | ~$130/10K 세션 | Photo Mechanic 대비 속도 차이 |
| CG-2: 태그 충돌 머지 UI 미노출 | Open | N/A | 사일런트 유니온 머지 리스크 |

"MEDIUM 이슈입니다"보다 "$130/10K 세션의 경쟁력 손실"이라고 하면 우선순위 판단이 훨씬 명확해집니다.

## 디자인 토큰 마이그레이션 사례

R44-R45에서 진행된 디자인 토큰 마이그레이션은 이 리뷰 시스템의 좋은 사례입니다.

### 변경 내용

```swift
// Before: 하드코딩된 색상
Text("Ready")
    .foregroundStyle(.green)

// After: 시맨틱 디자인 토큰
Text("Ready")
    .foregroundStyle(currentTheme.proSuccess)
```

UI-11에서 7개, UI-12에서 1개의 raw color literal을 시맨틱 토큰으로 교체했습니다.

### 4명의 리뷰어 반응

**Performance Engineer**: "토큰 읽기는 `ResolvedTheme` 구조체의 프로퍼티 접근. 힙 할당 없음, 새로운 관찰 의존성 없음. 성능 임팩트 제로."

**UI/UX Reviewer**: "UI-11, UI-12 수정 확인. `.green` → `proSuccess`, `Color.gray.opacity(0.2)` → `proTextQuaternary`. 시맨틱이 정확함."

**Photographer Reviewer**: "폴리시 작업이지 기능 진전은 아님. 점수 유지."

**Marketing Reviewer**: "사용자에게 보이는 변화 없음. GTM 영향 없음."

단 8줄의 변경이지만 4개의 다른 관점에서 검토되었습니다. 성능 리그레션이 없다는 확인, 시맨틱이 올바르다는 확인, 사용자 영향이 없다는 확인. 이런 검증을 혼자서 하려면 4가지 모자를 바꿔 써야 하는데, AI가 동시에 해줍니다.

## 멀티 플랫폼의 현실

BurstPick은 macOS와 Windows를 동시에 개발합니다. 둘 다 Swift 6.2지만 완전히 다른 세계에요.

| | macOS | Windows |
|---|---|---|
| UI | SwiftUI + AppKit | WinUI 3 |
| ML 추론 | CoreML + Metal | ONNX Runtime + DirectML |
| 이미지 디코딩 | CGImageSource, CIRAWFilter | WIC + D2D |
| 동시성 | `TaskGroup` + 구조화된 동시성 | 수동 옵저버 패턴 |
| 언어 수 | 39개 | 19개 |
| ML 모델 | 22개 | 서브셋 (DINOv2, RetinaFace 등) |

같은 사진 분석 로직이지만, 플랫폼별로 최적화 전략이 완전히 다릅니다. macOS에서는 Metal 컴퓨트 셰이더로 GPU 가속을 쓰고, Windows에서는 ONNX Runtime으로 CPU 기본 + DirectML 옵션을 제공합니다.

리뷰 시스템도 플랫폼 경계를 존중합니다. macOS 리뷰에서 Windows 코드를 지적하거나 그 반대는 허용되지 않아요. Ground Truth 문서에 "Windows is out-of-scope for macOS reviews"가 명시되어 있습니다.

## 22개 ML 모델의 관리

BurstPick이 복잡한 이유 중 하나는 22개의 ML 모델입니다:

- **화질 평가 5개**: Heuristic, TOPIQ-NR, MUSIQ, MANIQA, NIMA
- **미학 평가 2개**: LAION Aesthetic, ViT-B/16 Aesthetic
- **이미지 임베딩 3개**: Vision FeaturePrint, DINOv2, CLIP
- **얼굴 인식/임베딩 6개**: EdgeFace-XS/S, AdaFace IR-18/50, AuraFace, GhostFaceNets
- **VLM 평가 5개**: SmolVLM2 256M/2B, FastVLM 0.5B/1.5B, VLM Placeholder
- **분류 1개**: Apple Vision Classification

이 모델들의 점수를 가중합, 최대, 기하평균 등으로 조합해서 최종 순위를 매깁니다. 사용자가 ML Settings 패널에서 모델별 가중치를 조절할 수 있고, 실시간 미리보기로 결과를 확인합니다.

Performance Engineer 리뷰어가 이 파이프라인에서 가장 많이 지적하는 건 **메모리 압력(memory pressure)**입니다. 10,000장을 처리할 때 모든 모델을 메모리에 올려둘 수 없으니, 커널 레벨 메모리 모니터링으로 자동 캐시 퇴거와 모델 언로딩을 수행합니다.

## i18n: 39개 언어의 함정

39개 언어를 지원하면서 배운 교훈이 있습니다.

### 포맷 스트링 버그

```swift
// Before: 이렇게 쓰면 안 됨
String(localized: "\(count) tags")
// → 숫자가 로컬라이즈드 키에 삽입됨 (의도와 다름)

// After: C-스타일 포맷 사용
String(format: String(localized: "%lld tags"), Int64(count))
```

Swift의 `String(localized:)` 안에서 인라인 인터폴레이션을 쓰면, 런타임 값이 로컬라이제이션 키의 일부가 되어버립니다. `%lld` 포맷 스페시파이어로 바꿔야 번역가가 기대하는 C-표준 정수 포맷이 됩니다.

### 오염된 번역 제거

11개 로케일에서 2,971개의 크로스 언어 오염 번역을 발견하고 제거했습니다. 베트남어 "KEEP"(GIU)의 성조 표시가 빠져서 "GIỮ"로 수정하고, 루마니아어 "Reject" 번역이 의미 없는 문자열이었던 것도 고쳤습니다.

### CLDR 복수형 규칙

"1 photo" vs "2 photos" — 영어는 단수/복수지만, 아랍어는 0, 1, 2, few, many, other 6가지 형태가 있습니다. 39개 언어의 복수형 규칙(CLDR plural rules)을 모두 지원하는 건 리뷰 시스템 없이는 검증이 거의 불가능합니다.

## 신뢰와 데이터 안전성

Photographer Reviewer가 가장 중요하게 보는 것:

1. **RAW 파일은 절대 수정되지 않음** — 모든 결정은 카탈로그/XMP 메타데이터에 저장
2. **완전한 실행 취소** — 모든 컬링 결정은 세션 내에서 되돌릴 수 있음
3. **클라우드 의존성 없음** — 모든 처리가 로컬에서 수행

만약 BurstPick이 내일 사라져도, 데이터는 완전히 접근 가능합니다: SQLite 카탈로그(표준, 오픈 포맷), XMP 사이드카(Adobe 오픈 XML 표준), Lightroom 카탈로그 내보내기.

## 이 시스템을 가능하게 한 것

BurstPick의 AI 리뷰 시스템은 [agt](https://github.com/open330/agt)의 페르소나 기능 위에 구축되었습니다.

```bash
# 4명의 리뷰어가 동시에, 각각 다른 LLM으로
agt persona review security-reviewer --gemini \
  -o ".context/reviews/R45-security.md" &
agt persona review architecture-reviewer --codex \
  -o ".context/reviews/R45-architecture.md" &
agt persona review code-quality-reviewer --claude \
  -o ".context/reviews/R45-quality.md" &
agt persona review performance-reviewer --gemini \
  -o ".context/reviews/R45-performance.md" &
wait
```

결과는 `.context/reviews/`에 라운드 넘버링으로 저장됩니다. R1부터 R45까지의 히스토리가 있어서, 프로젝트의 품질이 어떻게 변화했는지 추적할 수 있습니다.

## 마치며

1인 개발자에게 코드 리뷰는 사치였습니다. 코드를 쓰는 사람과 리뷰하는 사람이 같으면, 자기가 놓친 문제를 자기가 발견하기 어렵죠.

AI 리뷰어는 이 문제를 구조적으로 해결합니다. 4명이 서로 다른 관점에서 동시에 리뷰하면, 보안 전문가는 OWASP를 보고 성능 전문가는 핫 패스를 보고 마케팅 전문가는 사용자 가치를 봅니다. 혼자서는 불가능한 다면적 검토가 가능해집니다.

45번의 라운드를 거치며 배운 가장 큰 교훈: **리뷰 시스템도 코드와 같이 진화해야 한다.** Ground Truth 문서, 심각도 체계, 달러 비용 추정, 점수 일관성 규칙 — 이 모든 것이 초기에는 없었고, 문제가 생길 때마다 하나씩 추가되었습니다.

AI가 인간 리뷰어를 완전히 대체할 수 있는지는 모르겠습니다. 하지만 적어도 1인 개발자에게, 리뷰어가 없는 것보다는 AI 리뷰어가 있는 것이 압도적으로 낫습니다. R45까지 온 BurstPick이 그 증거입니다.
