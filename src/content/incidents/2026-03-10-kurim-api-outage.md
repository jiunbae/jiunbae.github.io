---
title: "Kurim API Outage"
date: 2026-03-10T14:20:00+09:00
resolvedDate: 2026-03-10T16:45:00+09:00
severity: critical
status: resolved
affectedServices:
  - Kurim
published: true
timeline:
  - time: 2026-03-10T14:20:00+09:00
    status: investigating
    message: "Kurim API 응답 지연 및 5xx 오류 다수 감지."
  - time: 2026-03-10T14:45:00+09:00
    status: identified
    message: "FastAPI 서버의 메모리 누수로 인한 OOM 확인. 재시작 진행 중."
  - time: 2026-03-10T15:10:00+09:00
    status: monitoring
    message: "서버 재시작 완료. 메모리 사용량 모니터링 중."
  - time: 2026-03-10T16:45:00+09:00
    status: resolved
    message: "메모리 누수 패치 배포 완료. 정상 운영 확인."
---

## 사고 개요

2026년 3월 10일, Kurim API 서버에서 GPU 메모리 누수로 인한 OOM(Out of Memory)이 발생하여 약 2시간 25분간 API 응답 지연 및 5xx 오류가 발생했습니다.

## 영향 범위

| 항목 | 내용 |
|---|---|
| 영향 서비스 | Kurim (이미지 생성 API) |
| 장애 시간 | ~2시간 25분 (14:20 ~ 16:45 KST) |
| 영향받은 사용자 | Kurim API 사용자 전체 |
| 데이터 손실 | 없음 (실패한 요청은 재시도 가능) |

## 근본 원인 (Root Cause)

### 직접 원인
- FastAPI 서버의 이미지 생성 파이프라인에서 GPU 메모리가 정상적으로 해제되지 않는 버그 발생
- 누적된 메모리 사용으로 OOM 발생, 서버 프로세스 비정상 종료

### 근본 원인
1. 이미지 생성 파이프라인의 추론 완료 후 `torch.cuda.empty_cache()` 미호출
2. 에러 발생 시 GPU 텐서가 해제되지 않는 코드 경로 존재

## 조치 내역

### 즉시 조치
1. 서버 즉시 재시작으로 서비스 복구
2. 메모리 누수 원인이 된 코드 패치 배포

### 후속 조치 (Action Items)
- [x] 메모리 모니터링 알림 임계값 조정
- [x] GPU 메모리 누수 코드 패치 배포

## 교훈 (Lessons Learned)

1. **GPU 메모리 관리는 명시적으로 해야 한다.** Python GC에 의존하지 말고 추론 완료 및 에러 발생 시 반드시 `torch.cuda.empty_cache()`를 호출해야 한다.
