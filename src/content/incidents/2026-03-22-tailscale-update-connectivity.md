---
title: "Tailscale 업데이트로 인한 노드 간 연결 장애"
date: 2026-03-21T21:44:00+09:00
resolvedDate: 2026-03-22T01:02:00+09:00
severity: major
status: resolved
affectedServices:
  - Tailscale VPN
  - jiun-mini (K8s Host)
published: true
timeline:
  - time: 2026-03-21T21:44:00+09:00
    status: investigating
    message: "jiun-mini macOS Tailscale 앱이 1.96.2로 자동 업데이트. tailscaled 재시작으로 disco key 로테이션 발생."
  - time: 2026-03-21T21:44:00+09:00
    status: identified
    message: "gateway(1.92.5), s-lastorder(1.94.2) 등 구버전 노드에서 jiun-mini의 새 disco key를 DERP 릴레이에서 찾지 못함. 'derp does not know about peer [3Cwr7]' 반복 발생."
  - time: 2026-03-22T00:52:00+09:00
    status: identified
    message: "s-lastorder에서 jiun-mini SSH 연결 시도 타임아웃 (TCP 100.116.219.61:22). lastRecv=167h — 일주일 이상 통신 이력 없는 노드에서 key 전파 더욱 지연."
  - time: 2026-03-22T00:59:00+09:00
    status: monitoring
    message: "Tailscale 컨트롤 플레인에서 jiun-mini의 새 disco key 전파 완료. gateway, s-lastorder에서 disco key 변경 감지."
  - time: 2026-03-22T01:02:00+09:00
    status: resolved
    message: "s-lastorder → jiun-mini ping 성공 (192.168.32.55:41641 direct). 전체 Tailscale 메쉬 연결 정상 복구."
---

## 사고 개요

2026년 3월 21일 21:44경, jiun-mini의 macOS Tailscale 앱이 1.96.2로 자동 업데이트되면서 tailscaled가 재시작되었습니다. 이 과정에서 WireGuard disco key가 로테이션되었고, 구버전(1.92.5~1.94.2)을 실행 중인 다른 노드들이 DERP 릴레이를 통해 jiun-mini를 약 3시간 동안 찾지 못하는 장애가 발생했습니다.

## 영향 범위

| 항목 | 내용 |
|---|---|
| 영향 서비스 | Tailscale VPN 메쉬 네트워크 (jiun-mini 방향 통신) |
| 장애 시간 | ~3시간 18분 (21:44 ~ 01:02 KST) |
| 영향받은 노드 | gateway, s-lastorder → jiun-mini 연결 불가 |
| 데이터 손실 | 없음 |

## 근본 원인 (Root Cause)

### 직접 원인
- jiun-mini Tailscale 1.96.2 자동 업데이트 시 tailscaled 프로세스 재시작
- 재시작으로 disco key가 `2567d463eebc1000` → `b8016a8157b066f7`로 로테이션
- 구버전 노드들의 DERP 릴레이(derp-7 tok, derp-3 sin, derp-20 hkg)에서 새 키를 가진 jiun-mini를 인식하지 못함

### 근본 원인

1. **macOS Tailscale 자동 업데이트 미제어**: macOS App Store 또는 Tailscale 자체 자동 업데이트가 활성화되어 있어 사전 공지 없이 메이저 버전 업데이트가 적용됨
2. **노드 간 버전 불일치**: jiun-mini(1.96.2), s-lastorder(1.94.2), gateway(1.92.5) — 최대 4 마이너 버전 차이. 버전 간 disco protocol 변경으로 key 전파 지연 발생
3. **DERP 릴레이 캐시 만료**: 오래 통신하지 않았던 노드(s-lastorder lastRecv=167h)에서 key 전파가 더욱 지연됨
4. **LAN 환경에서도 DERP 의존**: 같은 서브넷(192.168.32.x)에 있음에도 disco key 갱신까지 direct path를 재설정하지 못함

## 조치 내역

### 즉시 조치
1. Tailscale 컨트롤 플레인에서 자연 key 전파 완료 대기 (00:59:30)
2. s-lastorder에서 `tailscale ping` 으로 direct path 재설정 확인 (01:02:38)

### 후속 조치 (Action Items)

- [ ] 전체 노드 Tailscale 버전 통일 (1.96.x 또는 최신 stable)
  - gateway: 1.92.5 → 업데이트 필요 (Ansible `tailscale` role 활용)
  - s-lastorder: 1.94.2 → 업데이트 필요
- [ ] jiun-mini macOS Tailscale 자동 업데이트 비활성화
- [ ] Tailscale 노드 상태 모니터링 추가 (disco key 변경, 노드 unreachable 감지)
- [ ] `tailscale set --auto-update=false` 적용 (자동 업데이트 비활성)

## 재발 방지 (Prevention)

### 1. Tailscale 자동 업데이트 비활성화

macOS Tailscale 앱의 자동 업데이트를 비활성화하여, 버전 업데이트를 유지보수 윈도우에서 수동으로 수행합니다:

```bash
# macOS (jiun-mini)
# Tailscale 앱 설정 > Auto-update 비활성화
# 또는 CLI로:
/Applications/Tailscale.app/Contents/MacOS/Tailscale set --auto-update=false

# Linux 노드 (gateway, s-lastorder)
# apt 자동 업데이트에서 tailscale 패키지 제외
sudo apt-mark hold tailscale
```

### 2. 전체 노드 버전 통일 정책

노드 간 Tailscale 버전 차이를 최대 1 마이너 버전 이내로 유지합니다. 업데이트 순서:

1. 비핵심 노드 먼저 업데이트 (s-lastorder 등)
2. 연결 확인 후 게이트웨이(subnet router) 업데이트
3. 마지막으로 핵심 서비스 노드(jiun-mini) 업데이트

### 3. Ansible Tailscale role에 버전 고정 추가

```yaml
# ansible/roles/tailscale/defaults/main.yml
tailscale_version: "1.96.2"  # 명시적 버전 고정
```

`state: present` 대신 특정 버전을 지정하여 의도치 않은 업데이트를 방지합니다.

### 4. 모니터링 강화

- Tailscale 노드의 `LastHandshake` 값을 주기적으로 체크하여, 특정 노드가 장시간 handshake 없는 상태를 감지
- Prometheus exporter 또는 스크립트로 `tailscale status --json`을 파싱하여 노드 상태 메트릭 수집

## 교훈 (Lessons Learned)

1. **자동 업데이트는 프로덕션 인프라의 적이다.** macOS 보안 업데이트(3/20 장애)에 이어 Tailscale 자동 업데이트까지, 자동 업데이트가 2주 만에 두 번째 장애를 유발했다. 모든 프로덕션 구성요소의 자동 업데이트를 비활성화하고, 변경을 IaC로 관리해야 한다.
2. **VPN 노드 버전 불일치는 시한폭탄이다.** 4 마이너 버전 차이(1.92→1.96)는 평시에는 문제없지만, disco key 로테이션 같은 이벤트에서 호환성 문제를 일으킨다. 버전을 통일하고, Ansible role에 버전을 고정해야 한다.
3. **LAN 노드라도 DERP 릴레이에 의존한다.** 같은 서브넷(192.168.32.x)에 있어도 Tailscale의 peer discovery는 DERP를 경유한다. 직접 통신 가능한 환경에서도 disco key 불일치 시 연결이 끊긴다.
