---
title: "Tokka 이메일 수신 장애 및 DB 연결 중단"
date: 2026-03-20T04:00:00+09:00
resolvedDate: 2026-03-20T18:35:00+09:00
severity: major
status: resolved
affectedServices:
  - Tokka
published: true
timeline:
  - time: 2026-03-18T18:11:00+09:00
    status: scheduled
    message: "macOS 26.3.1 (a) 보안 업데이트 자동 설치."
  - time: 2026-03-20T02:00:00+09:00
    status: investigating
    message: "macOS 자동 재부팅 실행. OrbStack VM 비정상 종료."
  - time: 2026-03-20T02:01:00+09:00
    status: investigating
    message: "시스템 부팅 후 OrbStack 자동 시작. containerd-shim 바이너리 exec format error 발생."
  - time: 2026-03-20T04:00:00+09:00
    status: identified
    message: "PostgreSQL/Redis pod RunContainerError 상태 진입. Tokka 앱은 기존 프로세스로 동작하나 DB 연결 불가."
  - time: 2026-03-20T15:00:00+09:00
    status: identified
    message: "이메일 워커 IMAP OVERQUOTA 에러 91회 누적 확인. 메일 수신 완전 중단 상태."
  - time: 2026-03-20T18:23:00+09:00
    status: monitoring
    message: "OrbStack 재시작으로 VM 복구. PostgreSQL/Redis pod 정상 기동."
  - time: 2026-03-20T18:26:00+09:00
    status: monitoring
    message: "이메일 워커 exponential backoff 패치 배포 (366ea6f). 폴링 60s, lookback 3일로 변경."
  - time: 2026-03-20T18:35:00+09:00
    status: resolved
    message: "전체 서비스 정상 확인. 이메일 backoff 동작 검증 완료."
---

## 사고 개요

2026년 3월 20일, macOS 보안 업데이트에 의한 자동 재부팅으로 OrbStack VM이 비정상 종료되어 containerd runtime이 손상되었습니다. 이로 인해 PostgreSQL과 Redis가 재시작 불가 상태에 빠졌고, 이메일 기반 분석 요청이 약 14시간 동안 처리되지 않았습니다.

## 영향 범위

| 항목 | 내용 |
|---|---|
| 영향 서비스 | Tokka (이메일 분석, 웹 분석 일부) |
| 장애 시간 | ~14시간 (02:01 ~ 18:35 KST) |
| 영향받은 사용자 | 이메일로 분석 요청한 사용자 (처리 지연) |
| 데이터 손실 | 없음 (메일은 Gmail에 보존, 복구 후 자동 처리) |

## 근본 원인 (Root Cause)

### 직접 원인
- OrbStack VM 내부 `containerd-shim-runc-v2` 및 `btrfs` 바이너리가 `exec format error`로 실행 불가
- PostgreSQL/Redis pod가 liveness probe 실패 → 재시작 시도 → `RunContainerError` 무한 루프

### 근본 원인
1. **macOS 26.3.1 (a) 보안 업데이트**가 3/18에 설치되고, 3/20 02:00에 자동 재부팅 실행
2. 자동 재부팅 시 OrbStack VM이 **graceful shutdown 없이 강제 종료**
3. VM 내부 btrfs 파일시스템 일부 손상 → containerd runtime 바이너리 읽기 오류
4. `orb restart` 명령도 같은 이유로 실패 (btrfs 유틸리티 자체도 실행 불가)

### 부가 원인: 이메일 워커
- Gmail IMAP 폴링이 30초 간격으로 반복되어 OVERQUOTA 한도 초과
- backoff 메커니즘이 없어 91회 연속 실패에도 동일 간격으로 재시도

## 조치 내역

### 즉시 조치
1. OrbStack 앱 종료 후 재실행 → VM 복구, 전체 pod 정상화
2. 이메일 워커 패치 배포 (`366ea6f`)
   - 폴링 간격 30s → 60s
   - OVERQUOTA 시 exponential backoff (60s → 120s → 240s → ... → 최대 900s)
   - lookback 기간 2일 → 3일 (장기 장애 시 메일 유실 방지)
   - `/api/config`에 `email_healthy` 상태 노출
   - 프론트엔드에 이메일 장애 경고 배너 추가

### 후속 조치 (Action Items)
- [ ] 프로덕션 워크로드를 로컬 OrbStack에서 클라우드/전용 서버로 이전 검토
- [ ] macOS 자동 업데이트 정책 변경 (수동 업데이트 또는 업데이트 시간 지정)
- [ ] K8s pod 상태 외부 모니터링 추가 (Uptime 체크 → 알림)
- [ ] OrbStack 자동 복구 스크립트 작성 (containerd 에러 감지 시 자동 재시작)

## 교훈 (Lessons Learned)

1. **로컬 환경에서 프로덕션을 운영하면 OS 업데이트가 곧 장애가 된다.** macOS의 자동 재부팅을 제어할 수 없는 한, VM 기반 인프라는 항상 이 리스크를 안고 있다.
2. **재시도 로직에는 반드시 backoff이 필요하다.** 고정 간격 재시도는 한도 기반 서비스(Gmail IMAP)에서 상황을 악화시킨다.
3. **장애 감지가 자동화되어 있지 않았다.** PostgreSQL이 14시간 동안 다운되었지만 알림이 없었다.
