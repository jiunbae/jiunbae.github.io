---
title: "서비스명 장애/작업 제목"
date: 2026-01-01T00:00:00+09:00
resolvedDate: 2026-01-01T02:00:00+09:00
severity: major # critical | major | minor | maintenance
status: resolved # investigating | identified | monitoring | resolved | scheduled
affectedServices:
  - ServiceName
published: false
timeline:
  - time: 2026-01-01T00:00:00+09:00
    status: investigating
    message: "최초 감지 내용."
  - time: 2026-01-01T00:30:00+09:00
    status: identified
    message: "원인 파악 내용."
  - time: 2026-01-01T01:00:00+09:00
    status: monitoring
    message: "조치 후 모니터링 내용."
  - time: 2026-01-01T02:00:00+09:00
    status: resolved
    message: "복구 완료 확인."
---

<!-- =============================================================
  Incident Report Template

  severity에 따라 적절한 포맷을 선택하세요:

  ■ maintenance / minor → 간결 포맷 (아래 Format A)
  ■ major / critical    → 전체 포맷 (아래 Format B)

  Format A 사용 시: "사고 개요" + "영향 범위" + "작업 내용" 만 작성
  Format B 사용 시: 아래 전체 섹션 작성
============================================================= -->

## 사고 개요

장애/작업의 한두 문장 요약. 무엇이, 언제, 얼마나 영향을 미쳤는지.

## 영향 범위

| 항목 | 내용 |
|---|---|
| 영향 서비스 | 서비스명 (구체적 기능) |
| 장애 시간 | ~N시간 (HH:MM ~ HH:MM KST) |
| 영향받은 사용자 | 범위 설명 |
| 데이터 손실 | 없음 / 설명 |

## 근본 원인 (Root Cause)

### 직접 원인
- 직접적으로 장애를 유발한 기술적 원인

### 근본 원인
1. 근본적인 원인 체인 (왜 → 왜 → 왜)

## 조치 내역

### 즉시 조치
1. 장애 복구를 위해 수행한 즉각적 조치

### 후속 조치 (Action Items)
- [x] 완료된 후속 조치
- [ ] 미완료 후속 조치

## 재발 방지 (Prevention)

### 1. 첫 번째 방지 대책 (적용 완료/예정)
구체적 설명

## 교훈 (Lessons Learned)

1. **핵심 교훈 제목.** 상세 설명.
