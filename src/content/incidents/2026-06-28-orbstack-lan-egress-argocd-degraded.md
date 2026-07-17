---
title: "OrbStack LAN egress 상실로 인한 prod ArgoCD 전수 Degraded"
date: 2026-06-28T04:40:00+09:00
resolvedDate: 2026-07-01T15:16:00+09:00
severity: major # critical | major | minor | maintenance
status: resolved # investigating | identified | monitoring | resolved | scheduled
affectedServices:
  - ArgoCD (central control plane)
  - Offsite backups (all prod apps)
  - Dev cluster GitOps sync
published: true
timeline:
  - time: 2026-06-28T04:40:00+09:00
    status: investigating
    message: "prod ArgoCD 앱들이 순차적으로 Degraded로 전환 시작(04:40~05:30 KST). dev 앱은 Unknown."
  - time: 2026-06-28T16:00:00+09:00
    status: investigating
    message: "prod 앱 14개 전수 Degraded 감지. 파드는 전부 Running/Available — 서비스가 아닌 health 표시 문제로 추정."
  - time: 2026-06-28T17:30:00+09:00
    status: identified
    message: "근본원인 격리: OrbStack VM/파드가 LAN(192.168.32.0/24) egress 상실. ArgoCD 컨트롤러가 dev(.66:6443) 도달 불가(i/o timeout·39회 재시작) + 오프사이트 백업 실패로 prod health가 Degraded로 동결. 인터넷 egress와 mac 호스트의 LAN 접근은 정상."
  - time: 2026-06-28T18:20:00+09:00
    status: monitoring
    message: "OrbStack 재시작으로 vmnet→LAN 경로 재구성. dev 16앱 재동기화(Healthy+Synced), 파드→LAN 정상, 컨트롤러 timeout 0. 단 오프사이트 백업 CronJob은 잔여 Degraded(백업 대상 autofs NFS 마운트 `/Volumes/nfs-share` wedged)."
  - time: 2026-07-01T15:16:00+09:00
    status: resolved
    message: "오프사이트 백업을 in-cluster 직접 NFS 마운트(.90:2049)로 재설계. 성공 실행 후 CronJob·앱 전부 Healthy. prod 22 + dev 16 = 38개 앱 Healthy+Synced, 오프사이트 Job 18개 Complete·실패 0 확인."
---

## 사고 개요

2026-06-28 새벽(~04:40 KST), prod 쿠버네티스 호스트인 **OrbStack(mac-mini, 192.168.32.55)의 VM/파드가 로컬 LAN(192.168.32.0/24)으로 나가는 egress 경로를 상실**했다. 인터넷 egress와 mac 호스트 자신의 LAN 접근은 정상이었으나, OrbStack 파드/노드만 LAN에 도달하지 못했다. 이로 인해 중앙 ArgoCD 컨트롤러가 dev 클러스터(192.168.32.66) API에 도달하지 못하고, prod 앱들의 오프사이트 백업이 실패하면서 **prod 앱 14개가 ArgoCD상 Degraded로 표시**됐다.

**실제 사용자 서비스는 전 기간 중단되지 않았다**(prod 파드는 계속 Running/Available). 영향은 관리 평면(ArgoCD)과 오프사이트 백업(DR)에 국한됐다. OrbStack 재시작으로 네트워크는 당일 복구했고, 잔존한 오프사이트 백업 문제(취약한 autofs NFS hostPath 마운트)는 백업을 in-cluster 직접 NFS 마운트로 재설계하여 07-01 최종 해소했다.

## 영향 범위

| 항목 | 내용 |
|---|---|
| 영향 서비스 | 중앙 ArgoCD(양 클러스터 관리), 오프사이트 백업(prod 전 앱), dev 클러스터 GitOps 동기화 |
| 장애 시간 | ~약 3일 (06-28 04:40 ~ 07-01 15:16 KST) — 네트워크는 06-28 저녁 복구, 백업은 07-01 복구 |
| 영향받은 사용자 | 없음 (사용자 대상 서비스 무중단; 내부 관리/DR 계층만) |
| 데이터 손실 | 없음 (로컬 백업 CronJob은 정상 지속; 오프사이트 사본만 ~3일 공백) |

## 근본 원인 (Root Cause)

### 직접 원인
- OrbStack의 prod 파드/VM이 LAN(192.168.32.0/24)으로의 egress를 상실. `pod → 192.168.32.66:6443`(dev API), `.65`, `.90` 등 LAN 대상이 모두 i/o timeout. 반면 `pod → 인터넷`과 `mac 호스트 → LAN`은 정상.
- ArgoCD application-controller(파드)가 dev 클러스터에 도달하지 못해 reconcile마다 30초 timeout이 누적되고 컨트롤러가 39회 재시작. 이 와중에 prod 앱들의 health 상태가 **Degraded로 동결**되어 이후 재계산으로도 갱신되지 않음.

### 근본 원인
1. **OrbStack의 vmnet→LAN 라우팅/NAT 상태 손상.** 파드망(flannel host-gw, cluster-cidr `192.168.194.0/25`)의 masquerade 규칙은 정상이었고, 트래픽은 OrbStack 게이트웨이(192.168.139.1)까지 도달한 뒤 LAN으로 포워딩되지 못했다. 즉 k8s 계층이 아니라 OrbStack VM↔호스트 LAN 네트워킹 계층의 문제였다. (다음 요인들은 조사 후 **원인에서 배제**: Tailscale 서브넷 라우트 `192.168.32/24→utun9`, macOS Internet Sharing `network_isolation`(LAN 미포함), `net.inet.ip.forwarding`.)
2. **prod 앱이 Degraded로 남은 실질 원인은 오프사이트 백업 CronJob.** ArgoCD 내장 CronJob 헬스체크는 "마지막 실행 실패 + 이후 성공 없음"(`lastScheduleTime > lastSuccessfulTime`)을 Degraded로 판정한다. dev 오버레이는 이 offsite CronJob을 삭제하므로 dev는 영향이 없었고, offsite CronJob을 가진 prod 앱 14개만 Degraded로 표시됐다.
3. **오프사이트 백업의 취약한 설계.** 백업 파드가 노드 hostPath `/Volumes/nfs-share`(macOS autofs로 마운트된 .90 NFS)를 OrbStack VM 미러를 통해 사용했다. 네트워크 단절 중 이 autofs 마운트가 wedged(stale handle) 상태가 됐고, SIP(automountd 재시작 차단)와 autofs(umount 불가) 때문에 SSH로는 복구할 수 없었다. (.90 NFS 서버 자체는 정상 — 수동 마운트는 성공.)

## 조치 내역

### 즉시 조치
1. 근본원인 격리: 파드→LAN egress 실패 확인(in-cluster/인터넷 대비), flannel masq·Tailscale·Internet Sharing·ip.forwarding 순차 배제.
2. **OrbStack 재시작**으로 vmnet→LAN 경로 재구성 → 파드→LAN 정상화, dev 클러스터 재연결(16앱 Healthy+Synced), 컨트롤러 timeout 해소.
3. 조사 중 임시 변경(ip.forwarding, Tailscale accept-routes)은 원인이 아님을 확인 후 **원상 복구**. outage 중 실패한 offsite 백업 Job 객체 정리.

### 후속 조치 (Action Items)
- [x] OrbStack 재시작으로 네트워크 근본원인 해소
- [x] 오프사이트 백업을 **in-cluster 직접 NFS 마운트(.90:2049)**로 재설계 — 취약한 autofs hostPath 미러 제거
- [x] 전 앱 Healthy + 오프사이트 백업 성공 검증
- [ ] OrbStack VM→LAN egress 상실의 정확한 트리거 규명(재발 시 조기 탐지 지표 마련)
- [ ] ArgoCD 컨트롤러 리소스 requests/limits 설정(현재 미설정) — 과부하 시 재시작 폭주 완화

## 재발 방지 (Prevention)

### 1. 오프사이트 백업 경로 견고화 (적용 완료)
hostPath → macOS autofs NFS → OrbStack VM 미러 경로를 폐기하고, 파드가 `.90:2049`로 직접 NFS를 마운트하도록 변경. macOS autofs/SIP/OrbStack-미러 의존성을 제거해 호스트 네트워크 이벤트에 대한 내성을 확보.

### 2. 조기 탐지 지표 (예정)
`pod → LAN` 도달성과 오프사이트 백업 성공 시각(`lastSuccessfulTime`)을 모니터링/알림에 추가. ArgoCD "전수 Degraded" 이전에 네트워크 단절을 잡을 수 있도록 한다.

## 교훈 (Lessons Learned)

1. **"전수 Degraded"는 앱이 아니라 관리 평면을 의심하라.** 모든 앱이 동시에 Degraded이고 파드는 Running이면, 개별 앱 문제가 아니라 ArgoCD 컨트롤러/네트워크 등 공통 상위 계층의 문제일 가능성이 높다. 실제로 파드는 전 기간 정상이었다.
2. **egress는 "인터넷 OK ≠ LAN OK".** OrbStack VM은 인터넷은 되면서 로컬 LAN만 못 나가는 상태였다. 도달성 점검은 in-cluster·인터넷·LAN을 분리해서 봐야 원인을 좁힐 수 있다.
3. **Degraded는 정확한 신호였다.** 표시를 억지로 지우는 대신, 그 원인(오프사이트 백업 실패 = 실재하는 DR 공백)을 해결하는 것이 옳았다.
4. **hostPath → OS 오토마운트 → VM 미러 체인은 취약하다.** macOS autofs NFS를 OrbStack 미러로 파드에 노출하는 방식은 호스트 네트워크 이벤트 한 번에 wedged가 되고 SIP 때문에 원격 복구가 막힌다. 상태 저장 의존성은 in-cluster 직접 마운트가 안전하다.
