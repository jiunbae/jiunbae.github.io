---
title: "Tailscale utun 인터페이스 누수 누적으로 인한 jiun-mini 인터넷 outbound 장애"
date: 2026-05-11T23:02:00+09:00
resolvedDate: 2026-05-12T02:02:00+09:00
severity: major
status: resolved
affectedServices:
  - jiun-mini (K8s Host)
  - OrbStack K8s Cluster
  - ArgoCD (repo-server)
published: true
timeline:
  - time: 2026-05-11T23:02:00+09:00
    status: investigating
    message: "jiun-mini 인터넷 outbound 장애 재발 보고. SSH(Tailscale)는 정상. 노드 uptime 50일 22시간."
  - time: 2026-05-11T23:08:00+09:00
    status: identified
    message: "ICMP/DNS는 정상이나 TCP 연결이 즉시 실패 (EADDRNOTAVAIL). utun 인터페이스 9개 누수 확인 (정상 1~3). Tailscale 데몬 IPC 단절 — `tailscaled.socket` 부재."
  - time: 2026-05-11T23:52:00+09:00
    status: monitoring
    message: "safe-reboot 절차로 OrbStack graceful stop → 재부팅. 부팅 후 Tailscale 1.96.2 → 1.96.5, macOS 26.3.1 → 26.4.1 업데이트. 인터넷 outbound 복구 (HTTP 200), utun 7개로 감소."
  - time: 2026-05-12T00:00:00+09:00
    status: identified
    message: "K8s 파드 헬스체크 중 `argocd-repo-server` pod이 2일 14시간째 CrashLoopBackOff 상태 발견 (이번 사건과 무관, 재부팅 이전 발생). init container `copyutil`의 `ln -s` 멱등성 결함."
  - time: 2026-05-12T02:02:00+09:00
    status: resolved
    message: "`kubectl rollout restart deployment argocd-repo-server`로 새 ReplicaSet 배포. 22초 만에 1/1 Running. 클러스터 전체 67 Running / 10 Completed / 0 Error. 모든 시스템 정상."
---

## 사고 개요

2026년 5월 11일 23시경, 자택 인프라의 prod K8s 워커 노드인 `jiun-mini`(Mac mini)에서 인터넷 outbound가 끊겼습니다. 같은 증상이 이전에도 발생한 **재발성 장애**로, 50일 22시간 무재부팅 동안 Tailscale macsys 빌드의 `utun` 인터페이스가 누수 누적되어 발생한 것으로 확인되었습니다. 재부팅 + Tailscale·macOS 업데이트로 복구했으며, 같은 점검 과정에서 별개의 ArgoCD `repo-server` 장애도 함께 발견·복구했습니다.

## 영향 범위

| 항목 | 내용 |
|---|---|
| 영향 서비스 | jiun-mini의 외부 인터넷 통신, OrbStack K8s에서 외부 API/이미지를 호출하는 워크로드, ArgoCD GitOps (repo-server) |
| 장애 시간 (인터넷) | ~50분 명시적 (23:02 ~ 23:52 KST) · 실제로는 수 시간~며칠에 걸쳐 점진 악화 |
| 장애 시간 (ArgoCD) | 2일 14시간 (5/9 ~ 5/12 02:02 KST, 재부팅 이전부터 누적) |
| 영향받은 사용자 | 외부 API/OAuth/이미지 풀이 필요한 K8s 워크로드 호출자. Tailscale·LAN 내부 통신은 정상 유지됨 |
| 데이터 손실 | 없음 |

## 근본 원인 (Root Cause)

### 직접 원인

- 50일 이상 무재부팅으로 macOS `utun` 가상 인터페이스가 9개까지 누적 (정상 1~3개)
- 그 결과 outbound TCP socket의 source 주소 선택이 실패 → `Can't assign requested address` (EADDRNOTAVAIL)
- ICMP/DNS는 source 주소 의존도가 낮아 정상 동작해 진단을 헷갈리게 함
- 부수적으로 Tailscale 데몬의 IPC 소켓이 끊겨 `tailscale status` 가 응답 불가 (`failed to connect to local Tailscale service`)

### 근본 원인

1. **Tailscale macsys 빌드의 utun 인터페이스 누수.** 시스템 확장(`io.tailscale.ipn.macsys.network-extension`)이 재시작·업데이트·자동 회복 시 이전 `utun` 인터페이스를 정리하지 않고 새 인터페이스를 추가로 생성. 50일 동안 시스템 확장이 여러 번 재시작되며 누적됨.
2. **장기 무재부팅.** 재부팅 주기가 없어 누수가 임계점(EADDRNOTAVAIL 발생 직전 ephemeral state)을 초과할 시간을 충분히 확보.
3. **클라이언트-서버 버전 mismatch.** 클라이언트 1.96.2, 컨트롤 1.96.5. 그 자체로는 정상 동작 범위지만 reconnect/restart 트리거가 잦아질 가능성.
4. **모니터링 부재.** `utun` 개수, `uptime`, Tailscale IPC 헬스에 대한 자동 알람이 없어 임계점 도달까지 능동 발견 불가.

### ArgoCD repo-server (별개 원인)

- Init container `copyutil` 명령:
  ```
  /bin/cp --update=none /usr/local/bin/argocd /var/run/argocd/argocd \
    && /bin/ln -s /var/run/argocd/argocd /var/run/argocd/argocd-cmp-server
  ```
- `cp --update=none`은 멱등(이미 있으면 skip)이지만 `ln -s`(no `-f`)는 **symlink가 잔존하면 `Already exists` 에러 → exit 1**
- `EmptyDir` 볼륨은 **pod 수명**을 따라가며 init container 재시작 시 살아남음 → 첫 init 성공 후 symlink 잔존 → 이후 init 무한 실패
- ArgoCD upstream의 init command 멱등성 결함 (`ln -s` → `ln -sf` 누락)

## 조치 내역

### 즉시 조치

1. **진단 캡처** (2026-05-11 23:02–23:10): SSH 가능 확인, `ifconfig`로 utun 9개 확인, `curl`로 EADDRNOTAVAIL 재현, Tailscale IPC 단절 확인
2. **runbook 작성** (`docs/jiun-mini/README.md`, `safe-reboot.sh`) — 재발에 대비한 절차 코드화
3. **safe-reboot 실행** (23:52): `orbctl stop` (graceful) → `sudo shutdown -r now`
4. **부팅 후 업데이트**: Tailscale 1.96.2 → 1.96.5, macOS 26.3.1 → 26.4.1 (별도 유지보수 창이 아닌 같은 사이클에 진행 — 인터넷 복구 검증과 분리 못 한 점은 트레이드오프)
5. **검증**: `curl https://1.1.1.1` HTTP 301, utun 7개, OrbStack `Running`, K8s 노드 `Ready`
6. **ArgoCD 복구** (02:02): `kubectl -n argocd rollout restart deployment argocd-repo-server` → 22초 만에 새 pod `1/1 Running`

### 후속 조치 (Action Items)

- [x] `docs/jiun-mini/` runbook 생성 (README + safe-reboot.sh)
- [x] Tailscale 1.96.5 적용
- [x] macOS 26.4.1 적용
- [x] ArgoCD repo-server 회복
- [ ] **월 1회 정기 재부팅 스케줄** (`sudo pmset repeat restartday MTWRFSU 04:00`) — 적용 여부 결정 필요
- [ ] **utun 개수·uptime 모니터링** — Prometheus exporter 또는 cron 스크립트로 임계치(utun ≥ 5, uptime > 30d) 알람
- [ ] **ArgoCD init container 멱등성 패치** — Helm values 또는 kustomize patch로 `ln -sf` 적용 (upstream PR 추적)
- [ ] **macOS 자동 업데이트 정책 점검** — 이번엔 사용자가 수동 진행했으나 가급적 자동 업데이트 비활성 + 유지보수 윈도우에서 수동 적용 (이전 2026-03-22 incident 교훈과 동일 방향)

## 재발 방지 (Prevention)

### 1. Runbook 코드화 (적용 완료)

진단·복구 절차를 IaC 레포 안에 저장하여 다음 재발 시 진단 시간을 단축합니다.

- `docs/jiun-mini/README.md` — 노드 개요, 알려진 이슈, 운영 절차(안전 재부팅 / Tailscale 업데이트 / macOS 업데이트), 헬스체크 빠른 참조
- `docs/jiun-mini/safe-reboot.sh` — 로컬에서 SSH로 실행하는 안전 재부팅 스크립트 (before/after snapshot + 다운→복귀→OrbStack 자동시작 대기까지 자동화)

### 2. 정기 재부팅 (검토 중)

```bash
# 매월 첫째 주 일요일 새벽 4시 (예시)
sudo pmset repeat restartday S 04:00
```

월 1회면 50일 누수 임계점 도달 전에 자연 해소 가능. 일일 재부팅은 과함 — 부팅 비용(OrbStack 콜드 스타트, K8s 파드 재배치)이 누적 위험을 상회.

### 3. 헬스체크 자동화 (계획)

다음 임계치 알람:

| 메트릭 | 임계 | 의미 |
|---|---|---|
| `ifconfig -l | grep -c utun` | ≥ 5 | utun 누수 의심 |
| `uptime` 일수 | > 30d | 재부팅 권장 |
| `curl https://1.1.1.1` HTTP code | `000` (1 min 연속) | outbound 단절 |
| `tailscale status` exit code | non-zero | Tailscale IPC 단절 |

수단: 노드 자체 cron + Tailscale DNS로 `gateway-1` 노드에 푸시, 또는 K8s 측 Prometheus → Alertmanager.

### 4. ArgoCD repo-server 패치 (계획)

upstream 차트의 init container 명령을 다음 중 하나로 패치:

```yaml
# 옵션 A: -sf 옵션 추가
command: ["sh", "-c"]
args:
  - |
    /bin/cp --update=none /usr/local/bin/argocd /var/run/argocd/argocd && \
    /bin/ln -sf /var/run/argocd/argocd /var/run/argocd/argocd-cmp-server
```

또는 단기적으로는 ArgoCD 차트 업그레이드 시점에 upstream에서 해결되었는지 확인.

## 교훈 (Lessons Learned)

1. **TCP만 실패하고 ICMP/DNS는 정상이면 source 주소 할당 실패를 의심하라.** `Can't assign requested address` (EADDRNOTAVAIL)는 라우팅 자체가 깨진 게 아니라 socket layer에서 막힌 것. macOS에서는 utun 인터페이스 누수가 흔한 트리거. `ping은 되는데 curl은 안 됨` 패턴을 보면 곧장 `ifconfig -l | grep -c utun`을 본다.
2. **macsys 빌드 Tailscale + 장기 무재부팅 = 시한폭탄.** 이전 2026-03-22 incident는 자동 업데이트로 인한 disco key 회전이 원인이었고, 이번엔 무재부팅으로 인한 utun 누수가 원인. 둘 다 macsys 시스템 확장의 lifecycle 문제. 정기 재부팅과 자동 업데이트 비활성화로 양쪽을 모두 차단해야 한다.
3. **K8s init container의 멱등성은 옵션이 아니라 필수다.** Pod sandbox는 재생성되지만 `EmptyDir`은 살아남는다. 그래서 init이 한 번 부분 성공하고 다른 컨테이너 문제로 sandbox가 재생성되면 init이 영원히 실패할 수 있다. `ln -sf`, `mkdir -p`, `rm -f` 같은 idempotent 변형을 항상 쓰자.
4. **운영 절차는 conversation에 두지 말고 코드로 남기자.** 같은 진단을 두 번째 하고 있다면 그건 첫 번째 진단을 잘못 저장한 것. 이번 사건을 계기로 `docs/jiun-mini/` runbook + safe-reboot 스크립트를 IaC 레포에 추가했다. 다음 재발은 SSH 후 한 줄로 진단되어야 한다.
5. **재부팅 + 업데이트를 같은 사이클에 묶지 말자 (이번엔 묶었음).** 인터넷 복구만이 목적이었다면 macOS 26.4.1까지 같이 적용하지 말았어야 했다. 새 macOS가 OrbStack/Tailscale 시스템 확장과 호환 문제를 일으켰다면 다운타임이 두 배가 됐을 것. 다행히 이번엔 무사했지만, runbook에 명시한 대로 "macOS 업데이트는 별도 유지보수 창" 원칙을 다음부터는 지키자.

## 후속 관찰 (2026-05-21)

복구 후 9일 동안 동일 노드를 관찰하면서, 초기 보고서의 근본 원인 가설을 수정해야 할 새 데이터를 확보했습니다.

### 관찰 결과

| 시점 | uptime | utun 개수 | 인터넷 | 비고 |
|------|--------|-----------|--------|------|
| 2026-05-11 23:02 (원 사건) | 50일 22시간 | 9 | ❌ EADDRNOTAVAIL | 임계 도달 |
| 2026-05-11 23:56 (재부팅 직후) | 4분 | 6 | ✅ | baseline |
| 2026-05-12 00:00 | 8분 | 7 | ✅ | +1 |
| 2026-05-21 17:03 | **2일 15시간** | **9** | ✅ | 다시 9개 도달, 그러나 인터넷 정상 |

### 가설 수정

**기존 가설**: "시스템 확장이 재시작될 때 옛 utun을 정리 안 함"
**수정 가설**: "시스템 확장이 **정상 운영 중에도** 새 utun을 할당하고 옛 utun fd를 close 안 함"

근거:
- 5/21 시점, Tailscale network-extension 프로세스(PID 1254)의 `etime`이 부팅 시각부터 한 번도 끊기지 않았음 (재시작 0회)
- 그럼에도 9개 utun 중 5개가 zombie (MTU 1380, IP 없음 — Tailscale 기본 MTU 시그니처)
- 활성 utun은 utun4 단 1개 (`100.116.219.61` 할당, MTU 1280)

누수 트리거 추정 (재시작 아님):
- Wi-Fi sleep/wake, 네트워크 인터페이스 전환
- Tailscale peer reconnect, NAT rebind, DERP 핸드오버
- 시스템 sleep/wake 사이클

매 이벤트마다 새 `NEPacketTunnelProvider` 인스턴스가 utun을 할당하고 이전 utun의 file descriptor를 release 안 함. 시스템 확장 프로세스 본인은 살아있으므로 OS도 fd를 회수 못 함.

### 추가 학습: utun 개수 ≠ EADDRNOTAVAIL 임계

5/21 시점에 utun이 9개인데 인터넷은 정상이라는 사실은, 원 사건의 직접 원인을 **utun 개수 단독**이 아닌 **utun 누수 + ephemeral port/소켓 누수 + 메모리 누수의 누적 조합**으로 봐야 함을 시사합니다. utun 개수는 누수의 *지표* (proxy metric)이지 임계 그 자체가 아닙니다.

→ runbook의 헬스체크 기준 `utun ≥ 5` 경고는 *조기 알람*으로는 유효하지만, 단독 임계로는 부적절. uptime 일수와 함께 봐야 합니다.

### 진짜 해결 방향

월 1회 재부팅(`pmset repeat restartday`)은 mitigation일 뿐 근본 해결이 아닙니다. 시스템 확장 누수는 macOS NetworkExtension 프레임워크의 알려진 패턴이고, Tailscale macsys 빌드를 쓰는 한 회피 불가능합니다.

근본 해결 옵션:

1. **subnet router 경유 (가장 깔끔)**: 같은 LAN의 다른 Tailscale 노드(`gateway-1` 등)가 `192.168.32.0/24`를 subnet route로 광고하도록 설정. jiun-mini에서 Tailscale 자체를 제거하고 `192.168.32.55` LAN IP로 접근. 시스템 확장 자체가 없으므로 누수 원천 차단.
2. **tailscaled CLI 데몬**: 오픈소스 `tailscaled` 바이너리를 launchd로 실행. NetworkExtension 미사용. macOS 공식 지원은 약하지만 기술적으로 가능.
3. **업스트림 버그 리포트**: Tailscale GitHub에 본 관찰 데이터(PID 재시작 없이도 utun 누수)와 함께 이슈 제출. 이전 reports는 "재시작 시 정리 안 됨" 가설에 머물러 있어 새 데이터 가치 있음.
4. **현상 유지 + 정기 재부팅**: 운영 비용 가장 낮음. 단 누수 속도(일일 ~2개)와 임계점이 매번 다를 수 있어 모니터링 필수.

prod 영향을 고려하면 옵션 1이 장기 정답. 옵션 4를 단기로 두면서 옵션 1 마이그레이션을 별도 작업 항목으로 추적합니다.
