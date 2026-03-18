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

Kurim API 서버에서 메모리 누수로 인한 장애가 발생했습니다.

## 원인

FastAPI 서버의 이미지 생성 파이프라인에서 GPU 메모리가 정상적으로 해제되지 않는 버그가 발견되었습니다.

## 조치

- 서버 즉시 재시작으로 서비스 복구
- 메모리 누수 원인이 된 코드 패치 배포
- 메모리 모니터링 알림 임계값 조정
