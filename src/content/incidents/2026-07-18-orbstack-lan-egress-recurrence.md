---
title: "OrbStack LAN egress 재상실로 인한 ArgoCD dev Unknown·prod Degraded (재발)"
date: 2026-07-18T04:00:00+09:00
resolvedDate: 2026-07-18T15:38:00+09:00
severity: major # critical | major | minor | maintenance
status: resolved # investigating | identified | monitoring | resolved | scheduled
affectedServices:
  - ArgoCD (central control plane)
  - Offsite backups (all prod apps)
  - Dev cluster GitOps sync
published: false
timeline:
  - time: 2026-07-18T04:00:00+09:00
    status: investigating
    message: "(사후 확인) 이 시각 정기 오프사이트 백업부터 실패 시작. 마지막 성공은 07-17 저녁(~16:20 KST). LAN egress 상실 추정 시점."
  - time: 2026-07-18T15:00:00+09:00
    status: investigating
    message: "정기 인프라 재점검 중 ArgoCD에서 dev 앱 17개 전수 Unknown, prod 앱 14개 Degraded 발견. 파드는 전부 Running — 서비스가 아닌 관리 평면 문제로 추정."
  - time: 2026-07-18T15:20:00+09:00
    status: identified
    message: "근본원인 격리: OrbStack pod·VM 호스트 양 계층에서 LAN(.64/.66/.90) 도달 불가, 인터넷만 정상. 06-28 사건의 재발. 컨트롤러가 dev(.66:6443) i/o timeout으로 reconcile 지연."
  - time: 2026-07-18T15:38:00+09:00
    status: monitoring
    message: "OrbStack 재시작(운영자)으로 vmnet→LAN 경로 재구성. dev 17앱 Synced+Healthy 복귀, pod→LAN 정상, 오프사이트 백업 수동 실행 성공(.90:2049 기록 확인), outage 중 실패한 offsite Job 18개 정리."
  - time: 2026-07-18T16:15:00+09:00
    status: resolved
    message: "서비스·워크로드 전 기간 정상, ArgoCD 40개 전부 Synced 확인. prod 14앱의 Degraded 표시는 offsite CronJob 헬스(마지막 정기 실행 실패)로, 다음 정기 백업(07-19 새벽 KST) 성공 시 자동 Healthy 예정."
---

## 사고 개요

2026-07-18, prod 쿠버네티스 호스트인 **OrbStack(mac-mini, 192.168.32.55)이 다시 로컬 LAN(192.168.32.0/24) egress를 상실**했다. 인터넷 egress와 mac 호스트 자신의 LAN 접근은 정상이었으나 OrbStack 파드/VM만 LAN에 도달하지 못했다. 그 결과 중앙 ArgoCD가 dev 클러스터(192.168.32.66) API에 도달하지 못해 **dev 앱 17개가 Unknown**, prod 앱들의 오프사이트 백업이 실패하며 **prod 앱 14개가 Degraded**로 표시됐다.

이는 [2026-06-28 동일 사건](/status/2026-06-28-orbstack-lan-egress-argocd-degraded)의 **재발**이다. 06-28 때 도입한 오프사이트 백업의 in-cluster 직접 NFS 마운트 재설계는 이번에도 정상 동작했으므로, 이번 원인은 백업 경로 취약성이 아니라 **OrbStack VM↔호스트 LAN 네트워킹의 근본적 재발**이었다. 마지막 정상 백업은 07-17 저녁(~16:20 KST)이었고, 07-18 새벽 정기 실행부터 실패했으나 **정기 재점검이 이뤄진 07-18 오후에야 감지**됐다. **사용자 대상 서비스는 전 기간 무중단**이었으며, OrbStack 재시작으로 네트워크는 당일 즉시 복구됐다.

## 영향 범위

| 항목 | 내용 |
|---|---|
| 영향 서비스 | 중앙 ArgoCD(dev 관리), 오프사이트 백업(prod 전 앱), dev 클러스터 GitOps 동기화 |
| 장애 시간 | ~약 11시간 30분 (07-18 04:00경 ~ 15:38 KST) — 발생은 새벽, 감지·복구는 오후 |
| 영향받은 사용자 | 없음 (사용자 대상 서비스 무중단; 내부 관리/DR 계층만) |
| 데이터 손실 | 없음 (로컬 백업 CronJob은 정상 지속; 오프사이트 사본만 ~1주기 공백) |

## 근본 원인 (Root Cause)

### 직접 원인
- OrbStack 파드·VM이 LAN(192.168.32.0/24) egress를 상실. `pod → .66:6443`(dev API), `.64`, `.90:2049`(백업 NFS) 등 LAN 대상이 모두 timeout. 반면 `pod → 인터넷(1.1.1.1)`과 `mac 호스트 → LAN`은 정상. **hostNetwork 파드 테스트로 CNI가 아닌 OrbStack VM 호스트 계층의 문제임을 확인**(VM은 LAN 트래픽을 게이트웨이 192.168.139.1로 올바르게 보내나, mac 호스트가 이를 LAN으로 포워딩/NAT하지 않음).
- ArgoCD application-controller가 dev 클러스터 도달 실패로 reconcile마다 30초 timeout을 누적, 컨트롤러가 dev 앱 resource tree를 갱신하지 못해 dev 앱이 Unknown으로 표시.

### 근본 원인
1. **OrbStack의 vmnet→LAN NAT/포워딩 상태 손상(재발).** 이번엔 두 계층 요인을 확인했다. (a) mac의 `net.inet.ip.forwarding`이 0으로 되돌아가 있었고(OrbStack이 기동 시 1로 설정하나 재부팅·타 도구가 0으로 리셋), (b) OrbStack이 자신의 LAN 마스커레이드(NAT) 규칙을 상실했다. **(a)만 1로 고쳐서는 복구되지 않았고, (b) 재설치를 위한 OrbStack 재시작이 필요**했다 — 06-28의 "vmnet NAT은 ip.forwarding에 의존하지 않는다"는 관찰과 일치한다.
2. **prod 앱이 Degraded로 표시된 실질 원인은 오프사이트 백업 CronJob.** ArgoCD 내장 CronJob 헬스체크는 `lastScheduleTime > lastSuccessfulTime`(마지막 정기 실행 실패 + 이후 성공 없음)를 Degraded로 판정한다. dev 오버레이는 이 offsite CronJob을 삭제하므로, offsite CronJob을 가진 prod 앱 14개만 Degraded로 표시됐다. `.status.resources`에는 CronJob 헬스가 비어(None) 표시되어 stuck처럼 오인하기 쉽다.
3. **조기 탐지 부재.** 06-28 액션 아이템으로 남겨둔 `pod→LAN` 도달성·오프사이트 백업 `lastSuccessfulTime` 알림이 아직 미구현이라, 새벽에 발생한 장애를 오후 정기 점검 때까지 감지하지 못했다.

## 조치 내역

### 즉시 조치
1. 근본원인 격리: `nc` 도달성 점검(in-cluster/인터넷/LAN 분리) → hostNetwork 파드로 VM 호스트 계층 확인 → mac 라우팅/pf/`ip.forwarding` 점검. Tailscale 서브넷 라우트(`accept-routes` 토글 무효, 원복)와 pf 차단 규칙은 원인에서 배제.
2. `net.inet.ip.forwarding=1` 설정(그것만으로는 미복구 확인).
3. **OrbStack 재시작**으로 LAN NAT 재설치 → pod→LAN 정상화, dev 17앱 재연결(Synced+Healthy), 컨트롤러 timeout 해소.
4. outage 중 실패한 offsite-sync Job 18개 정리, vaultwarden offsite-sync 수동 실행으로 `.90:2049` 기록 정상 복구 검증.

### 후속 조치 (Action Items)
- [x] OrbStack 재시작으로 네트워크 근본원인 해소, LAN egress 복구 검증
- [x] 실패 백업 Job 정리 + 오프사이트 백업 동작 재검증
- [ ] prod 14앱 Degraded 표시 자동 해소 확인(07-19 새벽 정기 백업 성공 시)
- [ ] **조기 탐지 지표 구현**(06-28에서 이월): `pod→LAN` 도달성 + offsite `lastSuccessfulTime` 알림
- [ ] OrbStack LAN egress 상실의 정확한 트리거·주기 규명(반복 재발 → 구조적 대책 필요)
- [ ] `net.inet.ip.forwarding=1` 부팅 시 자동 적용(LaunchDaemon) 검토

## 재발 방지 (Prevention)

### 1. 조기 탐지 지표 (예정, 최우선)
이번 재발도 새벽 발생을 오후에야 감지했다. `pod→LAN` 도달성 프로브와 오프사이트 백업 `lastSuccessfulTime` 지연을 모니터링/알림에 추가해, "전수 Degraded" 표시 이전 단계에서 네트워크 단절을 잡는다.

### 2. 구조적 대책 검토 (예정)
동일 장애가 06-28, 07-18 두 차례 재발했다. 노트북 위 OrbStack을 prod 관리 평면으로 두는 구조 자체가 LAN egress를 주기적으로 잃는다. `ip.forwarding` 영구화, OrbStack 네트워크 모드 재검토, 나아가 관리 평면 이전까지 포함해 근본 대책을 검토한다.

## 교훈 (Lessons Learned)

1. **"전수 Degraded"는 앱이 아니라 관리 평면/네트워크를 의심하라.** 06-28과 동일하게, 파드는 전 기간 Running이었고 문제는 OrbStack→LAN egress였다. in-cluster·인터넷·LAN을 분리한 도달성 점검이 원인을 빠르게 좁혔다.
2. **"인터넷 OK ≠ LAN OK", 그리고 계층 분리가 중요하다.** 이번엔 hostNetwork 파드 테스트로 CNI가 아닌 VM 호스트 계층임을 특정했다. `ip.forwarding=0`은 실재했으나 단독 원인이 아니었고, OrbStack 재시작(NAT 재설치)이 결정적이었다 — 한 증상에 멈추지 말 것.
3. **Degraded 표시를 지우려 컨트롤 플레인을 흔들지 마라.** 원인(offsite CronJob 헬스)을 확인하기 전에 ArgoCD controller·redis·repo-server·server를 반복 재시작했으나 무효였고, 오히려 OrbStack 컨테이너 이름 충돌로 롤아웃이 잠시 지연됐다. **`kubectl get cronjob …-offsite-sync -o json`으로 `lastScheduleTime` vs `lastSuccessfulTime`부터 확인**했다면 불필요한 조치였다. Degraded는 (지난번처럼) 정확한 신호였고, 원인 해소 후 다음 정기 실행에 스스로 회복한다.
4. **알려진 재발성 장애는 런북을 먼저 펼쳐라.** 06-28 보고서에 이번 원인·처방(OrbStack 재시작, CronJob 헬스 메커니즘)이 이미 문서화돼 있었다. 재점검 시 관련 사후분석을 먼저 확인하는 것이 진단 시간을 크게 줄인다.
