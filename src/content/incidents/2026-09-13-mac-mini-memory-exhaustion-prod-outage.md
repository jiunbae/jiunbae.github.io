---
title: "Mac mini 메모리 고갈로 OrbStack 붕괴, prod 전면 중단 (그리고 복구 중 자초한 2차 중단)"
date: 2026-09-13T16:35:00+09:00
resolvedDate: 2026-09-13T20:57:00+09:00
severity: critical
status: resolved
affectedServices:
  - 전체 prod 서비스 (26개, *.jiun.dev)
  - Cloudflare Tunnel (prod)
  - 중앙 ArgoCD
  - Prometheus / Alertmanager (in-cluster)
published: true
timeline:
  - time: 2026-09-12T21:00:00+09:00
    status: investigating
    message: "전조. Unity 배치 테스트 3개 동시 실행으로 load 50 도달, OrbStack 노드가 NotReady로 전환되며 vault.jiun.dev가 502. 부하가 가라앉자 자동 복구되어 일회성으로 판단했다. 같은 실패 유형의 첫 발현."
  - time: 2026-09-13T16:35:00+09:00
    status: investigating
    message: "운영자가 '모든 prod 서비스 접속 불가' 신고. 점검 결과 load 77, swap 14.2/15.4GB 소진, Kubernetes API가 connection refused. OrbStack 프로세스는 3초 전 재시작된 상태였다."
  - time: 2026-09-13T16:36:00+09:00
    status: identified
    message: "API 복귀. 그러나 78개 파드 중 준비된 것이 0개. kubectl은 Running으로 표시하지만 컨테이너 상태는 terminated인 좀비였다. cloudflared도 죽어 모든 도메인이 Cloudflare 530(터널 미연결)을 반환."
  - time: 2026-09-13T16:39:00+09:00
    status: monitoring
    message: "kubelet이 자동 복구 시작. 컨테이너가 30초 만에 28개에서 107개로 증가. 원인 프로세스 특정: ~/youtube 영상 파이프라인이 자식 26개 + ffmpeg + 헤드리스 Chrome 다수를 띄우고 있었다."
  - time: 2026-09-13T16:59:00+09:00
    status: identified
    message: "외부 접속 부분 복구. 그러나 cloudflared가 재시작 44회로 크래시 루프. 로그상 터널은 엣지 연결 4개를 정상 등록하는데도 파드가 죽고 있었다 — CPU limit 200m + readiness probe 3초/3회가 부하를 못 견딘 것."
  - time: 2026-09-13T17:15:00+09:00
    status: monitoring
    message: "cloudflared limit을 1000m/256Mi로, probe를 15초/8초/6회로 완화. 재시작 0회로 안정화. 남아 있던 jiun-api·bgm 파드 재기동 후 전 서비스 200 복귀."
  - time: 2026-09-13T19:30:00+09:00
    status: investigating
    message: "재발 방지를 위해 OrbStack 메모리를 16GB에서 8GB로 축소. prod 파드의 memory requests 합계가 11.2GiB라 8GB 노드에 담기지 않아 21개 파드가 Pending으로 떨어졌고, cloudflared가 포함되어 prod가 다시 530. 복구 조치가 2차 중단을 만들었다."
  - time: 2026-09-13T19:45:00+09:00
    status: monitoring
    message: "OrbStack을 12GB로 되돌려 Pending 해소, 서비스 복구. 이후 실사용 대비 requests 과선언(2.8배)을 실측 기준으로 재산정."
  - time: 2026-09-13T20:57:00+09:00
    status: resolved
    message: "공개 서비스 27개 전부 정상, 양 클러스터 비정상 파드 0개, ArgoCD 48개 Synced/Healthy. load 77 → 5.2, swap 14.2GB → 5.5GB. 알람을 s-10031로 이전 완료."
---

## 사고 개요

2026-09-13 오후, prod 쿠버네티스를 얹고 있는 **Mac mini(192.168.32.55)가 메모리 고갈로 OrbStack째 무너졌다.** 컨테이너가 전부 종료되면서 Cloudflare Tunnel을 물고 있던 `cloudflared`도 같이 죽었고, `*.jiun.dev` 전 도메인이 **Cloudflare 530(터널 미연결)** 을 반환했다.

전날인 09-12 밤에도 같은 유형의 장애가 있었다. 그때는 Unity 배치 테스트 3개가 원인이었고 부하가 가라앉자 스스로 복구되어 일회성으로 넘겼다. 이번엔 영상 생성 파이프라인이 방아쇠였고, 자동 복구되지 않았다.

부끄러운 대목이 하나 더 있다. **복구 후 재발 방지랍시고 OrbStack 메모리를 줄였다가 prod를 두 번째로 중단시켰다.** 경고는 스스로 해놓고 순서를 틀렸다.

그리고 이 장애는 **아무도 알림을 받지 못했다.** 운영자가 브라우저에서 직접 발견했다.

## 영향 범위

| 항목 | 내용 |
|---|---|
| 영향 서비스 | prod 전체 26개 서비스, Cloudflare Tunnel, 중앙 ArgoCD, in-cluster Prometheus·Alertmanager |
| 1차 중단 | 약 40분 (16:35 ~ 17:15 KST) — 전면 불가에서 점진 복구 |
| 2차 중단 | 약 15분 (19:30 ~ 19:45 KST) — 복구 조치가 자초 |
| 영향받은 사용자 | 전체 (사용자 대상 서비스 전면 중단) |
| 데이터 손실 | 없음 |

## 근본 원인

### 직접 원인

32GB 머신이 **56.3GB를 요구하고 있었다.** 압축기가 13.8GB를 점유한 채 50.9GB를 우겨넣었고(3.7배 압축), 그래도 모자라 스왑이 92% 찼다. 여유 메모리는 0.06GB.

이 상태에서는 어떤 프로세스든 페이지를 만질 때마다 압축 해제나 디스크 입출력이 발생한다. kubelet의 liveness/readiness probe가 타임아웃을 연발했고, OrbStack이 버티지 못하고 재시작했다.

### 메모리를 누가 쓰고 있었나

실측(`phys_footprint` 기준, RSS 아님)으로 뜯어보니 예상과 달랐다.

| 그룹 | 메모리 | 프로세스 |
|---|---|---|
| **Claude Code 세션** | **18.1 GB** | **58개** |
| OrbStack (prod k8s) | 13.1 GB | 2 |
| node | 10.6 GB | 51 |
| Codex | 4.3 GB | 39 |
| Chrome/Playwright | 3.5 GB | 34 |
| 기타 | 4.8 GB | 114 |

가장 큰 소비자가 prod 쿠버네티스가 아니라 **에이전트 CLI 세션 58개**였다. 그중 49개가 3일 이상, 15개가 17일째 살아 있었다. 전체 프로세스 1,172개 중 530개가 14일 초과였다.

세션이 끝나도 프로세스가 죽지 않는다. zsh 256개, muxa 77개, gitstatusd 65개, node 50개가 그 잔해다.

### 왜 알림이 안 갔나

**Prometheus와 Alertmanager가 둘 다 감시 대상인 Mac mini 안에서 돌고 있었다.** 호스트가 죽으면 알람도 같이 죽는다. PrometheusRule 31개와 Telegram 수신처까지 멀쩡히 설정돼 있었지만 아무 소용이 없었다.

s-10031의 LXC Prometheus는 장애 내내 살아 있었다. 그런데 `rule_files`와 `alerting` 설정이 아예 없고 Alertmanager도 없었다. 룰 0개. 살아남은 쪽은 볼 줄 몰랐고, 볼 줄 아는 쪽은 같이 죽었다.

### 2차 중단은 왜

메모리 압박을 줄이려고 OrbStack 할당을 16GB에서 8GB로 낮췄다. prod 실사용이 4.2GiB니 충분하다고 봤다.

**쿠버네티스는 실사용이 아니라 requests로 스케줄링한다.** 선언된 합계가 11.2GiB라 8GB 노드에는 들어가지 않았고, 21개 파드가 Pending으로 떨어졌다. 하필 cloudflared가 거기 끼어 있었다.

적용 전에 이 위험을 인지하고 경고까지 했으면서도, requests를 먼저 정리하지 않고 진행했다. 순서를 바꿨으면 없었을 중단이다.

## 조치 내역

### 즉시 조치

1. 고아 프로세스 10개 종료. RSS로는 46MB였지만 압축분 포함 **실제 8.3GB**였다. 53일 된 `omp ingest --stdin` 3개, 17일 된 expo dev server 등 잊힌 개발 서버들.
2. cloudflared CPU limit 200m → 1000m, memory 128Mi → 256Mi, readiness probe 3초/3회 → 8초/6회. 재시작 44회에서 0회로.
3. 오래된 tmux 세션 4개와 포트를 잡고 있던 개발 서버 4개 정리.
4. OrbStack 12GB로 원복 후 Pending 해소.

### 구조 개선

1. **알람을 s-10031로 이전.** Alertmanager 0.34.0을 monitoring LXC에 올리고, 호스트 레벨 룰 6개(TargetDown, ProdClusterUnreachable, HostHighLoad, HostMemoryLow, HostSwapExhausted, HostDiskLow)를 붙였다. 임계치는 이번 장애 실측값에서 뽑았다 — load5가 코어 수의 7.7배, 스왑 96%. Telegram 발송까지 테스트로 확인했다.
2. **requests 실측 기준 재산정.** 10.6 GiB → 7.3 GiB. 과선언 2.8배를 걷어냈다.

| 네임스페이스 | 전 | 후 | 실사용 |
|---|---|---|---|
| kongbu | 2016 Mi | 1056 Mi | 738 |
| ssudam | 1632 Mi | 1024 Mi | 556 |
| bubbles | 896 Mi | 288 Mi | 179 |
| swiq-server | 448 Mi | 224 Mi | 135 |
| jiun-api | 448 Mi | 224 Mi | 142 |
| oh-my-prompt | 576 Mi | 224 Mi | 86 |

3. 미사용 서비스 정리: kurim, selectchatgpt 종료(둘 다 데이터 0건 수준, 최종 덤프는 NFS에 보관).

### 남은 과제

- [ ] OrbStack 8GB 재시도 (requests 정리로 이제 가능)
- [ ] 오래된 tmux 세션 36개 정리 (17일 이상 13개, 약 20GB)
- [ ] 무거운 호스트 작업의 우선순위 낮추기 (`nice`, `taskpolicy`)
- [ ] prod를 워크스테이션에서 분리 — 근본 대책이나 하드웨어가 필요하다

## 교훈

### 1. RSS를 믿지 마라

고아 프로세스 10개의 RSS 합계는 46MB였다. 실제 점유는 8.3GB. **180배 차이다.**

macOS는 유휴 페이지를 압축기와 스왑으로 밀어낸다. 그러면 RSS에서 빠지지만 메모리는 계속 잡고 있다. 오래 살아 있는 유휴 프로세스일수록 RSS가 실제를 심하게 과소평가한다. `phys_footprint`로 봐야 한다.

`ps` 출력만 보고 "메모리 별로 안 쓰는데?"라고 넘어갔다면 범인을 영영 못 찾았다.

### 2. 감시자를 감시 대상 안에 두지 마라

이게 가장 값비싼 교훈이다. 알람 인프라를 정성껏 구성해놓고 그것을 감시 대상 호스트에 얹으면, 정작 필요한 순간에 침묵한다.

살아남는 계층에 최소한의 룰이라도 두는 것이 화려한 대시보드보다 낫다. s-10031의 Prometheus는 계속 돌고 있었지만 룰이 없어서 아무 말도 못 했다.

### 3. 복구 조치가 2차 장애를 만든다

메모리를 줄이면 좋다는 것까지는 맞았다. 하지만 쿠버네티스가 requests로 스케줄링한다는 사실을 계산에 넣지 않았다. **위험을 인지하고 경고까지 했으면서 순서를 바꾸지 않았다.**

장애 대응 중에는 판단이 급해진다. "이것만 하면 나아진다"는 조치일수록 적용 전에 한 번 더 멈춰야 한다. 특히 이미 불안정한 시스템에서는.

### 4. 세션은 끝나도 죽지 않는다

에이전트 CLI 세션 하나가 유휴 상태로 250~400MB를 무기한 붙들고 있다. 측정해보니 그중 83%가 JavaScript 힙이고, JIT 코드 22MB는 프로세스마다 따로 갖는다. 공유율은 4.1%에 불과했다.

세션 하나로는 문제가 아니다. 58개면 18GB다. 2주마다 이 상황이 반복될 구조였다.

도구의 메모리 설계를 탓하기 전에, **끝난 세션을 정리하는 습관이 먼저**다. 그리고 그것을 사람의 기억에 맡기지 말고 자동화해야 한다.

### 5. 전조를 흘려보내지 마라

09-12 밤 vault.jiun.dev가 502로 죽었을 때, 부하가 가라앉자 자동 복구되기에 일회성으로 넘겼다. 24시간도 안 되어 같은 원인으로 훨씬 크게 터졌다.

"저절로 나았다"는 것은 원인이 사라졌다는 뜻이 아니다. 조건이 잠시 완화됐을 뿐이다.
