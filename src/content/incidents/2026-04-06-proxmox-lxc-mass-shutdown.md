---
title: "Proxmox LXC/VM 일괄 Shutdown 및 삭제 장애"
date: 2026-04-06T11:22:00+09:00
resolvedDate: 2026-04-06T12:59:00+09:00
severity: major
status: resolved
affectedServices:
  - Gateway (CoreDNS)
  - Gateway (Nginx Reverse Proxy)
  - Monitoring (Prometheus/Grafana)
  - Nextcloud
  - Docker Registry
  - k3s-dev Cluster
  - Obsidian Sync (CouchDB)
published: true
timeline:
  - time: 2026-04-06T11:22:23+09:00
    status: investigating
    message: "192.168.32.55(Mac Mini)에서 양쪽 Proxmox 서버로 Proxmox API 일괄 호출 발생. s-10031: LXC 110,120,130,140,150 shutdown + 130,150 destroy + 170 reboot. s-lastorder: VM 200 shutdown + VM 100 reboot."
  - time: 2026-04-06T12:38:00+09:00
    status: identified
    message: "sisters.internal.jiun.dev SSH 접속 불가 신고. DNS(*.internal.jiun.dev) 해석 실패 확인. Gateway LXC(110) stopped 상태 확인."
  - time: 2026-04-06T12:43:00+09:00
    status: monitoring
    message: "Gateway LXC(110) 시작 완료. CoreDNS, Nginx 정상 기동 확인. DNS 해석 및 웹 접속 복구 확인."
  - time: 2026-04-06T12:47:00+09:00
    status: monitoring
    message: "Monitoring LXC(120), Nextcloud LXC(140) 시작 완료. Prometheus, Grafana 정상 기동 확인."
  - time: 2026-04-06T12:52:00+09:00
    status: monitoring
    message: "k3s-dev VM(200) 시작 완료. k8s-master 노드 Ready 상태 확인."
  - time: 2026-04-06T12:59:00+09:00
    status: resolved
    message: "복구 가능한 모든 서비스 정상화 확인. Registry LXC(130)과 Obsidian Sync LXC(150)은 destroy되어 재생성 필요."
---

## 사고 개요

2026-04-06 11:22 KST, Mac Mini(192.168.32.55)에서 양쪽 Proxmox 서버(s-10031, s-lastorder)에 대해 Proxmox REST API를 통한 일괄 shutdown/destroy 작업이 실행되었다. Gateway(CoreDNS, Nginx), Monitoring, Nextcloud, Registry, k3s-dev 클러스터 등 주요 인프라 서비스가 약 1시간 37분간 중단되었으며, Registry LXC(130)와 Obsidian Sync LXC(150)은 완전 삭제되었다.

## 영향 범위

| 항목 | 내용 |
|---|---|
| 영향 서비스 | CoreDNS(`*.internal.jiun.dev` 해석), Nginx 리버스 프록시, Prometheus/Grafana, Nextcloud, Docker Registry, k3s-dev 클러스터, Obsidian Sync |
| 장애 시간 | ~1시간 37분 (11:22 ~ 12:59 KST) |
| 영향받은 사용자 | 내부 서비스 전체. DNS 해석 불가로 `*.internal.jiun.dev` 도메인 접속 불가 |
| 데이터 손실 | Registry LXC(130, 192.168.32.30)와 Obsidian Sync LXC(150, 192.168.32.50) 완전 삭제. 컨테이너 이미지 및 CouchDB 데이터 손실 가능 (외부 스토리지 마운트 여부 확인 필요) |

## 근본 원인 (Root Cause)

### 직접 원인

Mac Mini(192.168.32.55)에서 Proxmox REST API(`POST /api2/json/access/ticket` → `POST .../status/shutdown` → `DELETE .../lxc/{vmid}`)를 호출하여 LXC/VM을 일괄 종료 및 삭제했다.

### 근본 원인

1. Mac Mini에서 다수의 Claude Code 세션이 `--dangerously-skip-permissions` 모드로 실행 중 (10개 이상 동시 실행)
2. 이 중 하나의 프로세스가 Proxmox REST API를 호출하여 파괴적 작업을 수행. Claude Code 세션 로그, 셸 히스토리, 프로젝트 소스 코드를 전수 조사했으나 **정확한 출처 세션은 특정 불가** — 동적 생성된 curl/Python 스크립트를 통한 간접 호출로 추정
3. API 호출 패턴(LXC 생성 6회 시도 → 500 에러 → 기존 LXC 일괄 shutdown → destroy)은 인프라를 재구성하려다 실패 후 정리를 시도한 자동화 에이전트 행동과 일치
4. Proxmox API 접근이 root@pam 크레덴셜로 인증 제한 없이 가능한 상태
5. 파괴적 API 호출(DELETE)에 대한 보호 메커니즘(확인 절차, ACL 제한) 부재

### API 호출 시퀀스 (s-10031 access log)

```
11:22:23 POST /api2/json/access/ticket                    → 200 (인증)
11:22:23 POST /api2/json/nodes/s-10031/lxc                → 500 ×6 (LXC 생성 시도 실패)
11:22:23 POST .../lxc/140/status/shutdown                  → 200
11:22:23 POST .../lxc/120/status/shutdown                  → 200
11:22:23 POST .../lxc/130/status/shutdown                  → 200
11:22:23 POST .../lxc/150/status/shutdown                  → 200
11:22:23 POST .../lxc/110/status/shutdown                  → 200
11:22:23 POST .../lxc/170/status/reboot                    → 200
11:22:29 DELETE /api2/json/nodes/s-10031/lxc/130           → 200
11:22:30 DELETE /api2/json/nodes/s-10031/lxc/150           → 200
```

동시에 s-lastorder에서도:
```
11:22:23 POST .../qemu/200/status/shutdown                 → 200
11:22:24 POST .../qemu/100/status/reboot                   → 200
```

## 조치 내역

### 즉시 조치

1. Gateway LXC(110) 시작 → CoreDNS, Nginx 복구 (12:43)
2. Monitoring LXC(120) 시작 → Prometheus, Grafana 복구 (12:47)
3. Nextcloud LXC(140) 시작 → Nginx 복구 (12:47)
4. k3s-dev VM(200) 시작 → k8s-master Ready 확인 (12:52)
5. 전체 DNS 해석 및 웹 접속 복구 확인 (12:59)

### 후속 조치 (Action Items)

- [x] 복구 가능한 LXC/VM 전체 시작
- [x] 서비스 상태 전수 점검
- [x] 장애 원인 분석 (API 호출 출처 추적)
- [ ] Registry LXC(130) 재생성 — Ansible role 활용
- [ ] Obsidian Sync LXC(150) 재생성 필요 여부 판단
- [ ] Proxmox API 접근 제한 강화 (API 토큰 분리, 읽기 전용 토큰 발급)
- [ ] 파괴적 API 호출에 대한 알림/차단 메커니즘 도입
- [ ] Claude Code 세션의 `--dangerously-skip-permissions` 사용 검토

## 재발 방지 (Prevention)

### 1. Proxmox API 접근 제어 (미적용)
- root@pam 크레덴셜 대신 제한된 권한의 API 토큰 발급
- 파괴적 작업(DELETE, shutdown)에 대한 별도 권한 분리
- 모니터링/대시보드용 읽기 전용 토큰과 관리용 토큰 분리

### 2. LXC `protection` 플래그 활성화 (미적용)
- 주요 LXC에 `protection: 1` 설정으로 실수로 인한 삭제 방지

### 3. Claude Code 권한 관리
- `--dangerously-skip-permissions` 대신 적절한 권한 모드 사용 검토
- 인프라 관리 세션의 Proxmox 크레덴셜 접근 범위 제한

## 교훈 (Lessons Learned)

1. **자동화 에이전트에 파괴적 권한을 부여하면 예측 불가능한 결과가 발생할 수 있다.** Claude Code 세션이 LXC 생성을 시도(500 에러)한 후 기존 LXC를 shutdown/destroy하는 패턴을 보였다. AI 에이전트에게 인프라 관리 권한을 부여할 때는 읽기/쓰기/삭제 권한을 명확히 분리해야 한다.

2. **Gateway(DNS)가 단일 실패점(SPOF)이었다.** Gateway LXC가 중단되자 `*.internal.jiun.dev` 전체 도메인 해석이 불가능해졌다. 이중화 또는 폴백 DNS 구성이 필요하다.

3. **onboot 설정만으로는 충분하지 않다.** 모든 LXC에 `onboot: 1`이 설정되어 있었지만, API를 통한 명시적 shutdown에는 보호가 되지 않았다. `protection` 플래그와 API 레벨 접근 제어가 함께 필요하다.
