---
title: "OrbStack VM containerd 폭주로 인한 Kubernetes 전체 장애"
date: 2026-06-24T17:00:00+09:00
severity: major
status: monitoring
affectedServices:
  - OrbStack Docker Runtime
  - OrbStack K8s Cluster
  - Vaultwarden
  - Cloudflare Tunnel
published: true
timeline:
  - time: 2026-06-24T17:00:00+09:00
    status: investigating
    message: "Mac 호스트 load average가 18까지 상승하고, kubectl/docker 명령이 TLS 핸드셰이크 타임아웃. vault.jiun.dev는 Cloudflare 1033 tunnel_error로 접속 불가."
  - time: 2026-06-24T17:30:00+09:00
    status: identified
    message: "OrbStack VM 내부 containerd가 망가진 sandbox를 반복 재생성하면서 seccomp BPF JIT, text_poke, 전 vCPU IPI 동기 대기를 유발한 것으로 확인."
  - time: 2026-06-24T17:45:00+09:00
    status: identified
    message: "vCPU를 10개에서 4개로 줄이고 k8s 자동 기동을 비활성화. 워치독 cron도 임시 중단해 복구 중 자동 재기동을 차단."
  - time: 2026-06-24T18:31:00+09:00
    status: monitoring
    message: "기존 OrbStack data.img.raw를 APFS clone 방식으로 백업. 논리 크기 8.0T, 실제 할당량 62G. PVC/DB 데이터는 이 이미지 안에 보존."
  - time: 2026-06-24T18:34:00+09:00
    status: monitoring
    message: "orbctl reset 후 fresh Docker engine이 정상 응답. k8s.enable=false 유지. 서비스와 PVC 데이터 복원은 후속 작업으로 분리."
---

## 사고 개요

2026년 6월 24일, OrbStack VM 내부의 `containerd`가 망가진 container sandbox를 계속 재생성하면서 Mac 호스트와 Kubernetes 전체가 동시에 멈췄습니다. 표면적으로는 `vault.jiun.dev`가 Cloudflare 1033 에러를 내고, `docker`와 `kubectl` 명령이 타임아웃되는 장애였습니다.

조금 더 골치 아팠던 점은 재부팅으로 해결되지 않았다는 것입니다. 깨진 containerd 상태가 OrbStack VM 디스크에 남아 있어서, VM이 다시 뜨면 같은 sandbox 재생성 루프가 반복됐습니다. 결국 기존 VM 디스크를 통째로 백업한 뒤 `orbctl reset`으로 런타임 상태를 초기화했습니다.

## 영향 범위

| 항목 | 내용 |
|---|---|
| 영향 서비스 | OrbStack Docker Runtime, OrbStack K8s Cluster, Vaultwarden, Cloudflare Tunnel 기반 서비스 |
| 장애 시간 | 2026-06-24 17:00 KST경 인지. 18:34 KST에 Docker runtime은 정상화. K8s 서비스 재배포와 PVC 데이터 복원은 별도 후속 작업 |
| 영향받은 사용자 | 내부 homelab 서비스 사용자. 특히 vault.jiun.dev 접속과 K8s 기반 서비스 전체 |
| 데이터 손실 | 현재 확인된 손실 없음. 다만 기존 PVC 데이터는 fresh OrbStack에 복원된 상태가 아니라 백업 raw 이미지 안에 보존된 상태 |

## 근본 원인 (Root Cause)

### 직접 원인

- OrbStack VM 내부의 `containerd`가 손상된 container sandbox를 고빈도로 재생성
- 매 재생성 시 seccomp BPF 필터 JIT 컴파일과 Linux kernel `text_poke` 경로 진입
- `kick_all_cpus_sync`, `synchronize_rcu_expedited`가 모든 vCPU에 IPI를 보내고 동기 대기
- vCPU 10개 구성이 Apple Virtualization host thread 전체를 끌고 들어가면서 Mac 호스트까지 load storm 발생

`~/.orbstack/log/vmgr.log`에는 `containerd`가 RCU stall을 일으키는 커널 스택이 남아 있었습니다.

```text
CPU: 1  PID: 250  Comm: containerd   task:rcu_sched state:R running task
rcu: INFO: rcu_sched self-detected stall on CPU  (t=370024 jiffies, ncpus=10)

do_seccomp
 -> bpf_prog_create_from_user
 -> bpf_int_jit_compile
 -> bpf_arch_text_copy
 -> __text_poke
 -> kick_all_cpus_sync
 -> smp_call_function_many_cond
```

### 근본 원인

1. **containerd 상태 영속화.** 깨진 sandbox 상태가 VM 디스크에 저장되어, VM 재시작 후에도 같은 재생성 루프가 반복됐다.
2. **vCPU 과다 구성.** vCPU가 10개라 seccomp JIT의 cross-CPU synchronization 비용이 커졌고, 문제가 VM 내부에 머물지 않고 Mac 호스트 행으로 번졌다.
3. **로컬 PV/백업 의존.** 모든 DB와 백업이 OrbStack VM 내부 `local-path` PVC에 있어, 단순 reset이 데이터 손실로 이어질 수 있었다.
4. **워치독 타임아웃 부재.** 기존 워치독은 `docker ps`가 멈췄을 때 실패하지 않고 무한 대기했고, cron이 5분마다 새 인스턴스를 쌓았다.

## 조치 내역

### 즉시 조치

1. OrbStack vCPU를 10개에서 4개로 축소해 IPI storm이 호스트 전체를 잠그는 정도를 낮춤
2. `k8s.enable=false`로 Kubernetes 자동 기동 비활성화
3. 워치독 cron 임시 중단
4. 기존 `data.img.raw`를 APFS clone 방식으로 백업
5. `orbctl reset`으로 깨진 containerd 상태 제거
6. fresh Docker engine 정상 응답 확인

백업 위치:

```text
~/OrbStack-backups/2026-06-24-containerd-storm/data.img.raw
```

백업 이미지는 논리 크기 `8.0T`, 실제 할당량 `62G`인 sparse raw 파일입니다. 실수로 덮어쓰지 않도록 read-only 권한으로 바꿔 두었습니다.

### 후속 조치 (Action Items)

- [x] vCPU 10 -> 4 축소
- [x] 워치독 스크립트 하드닝 (`timeout 15`, 중복 실행 방지, graceful -> SIGKILL fallback)
- [x] k8s 자동 기동 비활성화
- [x] 워치독 cron 임시 중단
- [x] 기존 VM 디스크 이미지 백업
- [x] OrbStack reset 및 fresh Docker engine 확인
- [ ] K8s 재활성화와 GitOps 재배포
- [ ] 필요한 PVC 데이터만 백업 이미지에서 추출해 복원
- [ ] Vaultwarden 데이터 복구 경로 최종 확인
- [ ] 외부 백업 도입 (NFS/S3/restic 등)
- [ ] 워치독 cron 재활성화

## 재발 방지 (Prevention)

### 1. 외부 백업 도입

이번 장애의 가장 큰 문제는 runtime reset보다 데이터였습니다. DB와 백업이 모두 같은 OrbStack VM 내부 `local-path` PVC에 있었기 때문에, VM을 reset하면 백업까지 같이 사라지는 구조였습니다.

앞으로는 백업 CronJob이 `local-path`가 아니라 외부 저장소로 나가야 합니다. 후보는 NFS(`192.168.32.90`), S3 호환 스토리지, restic 같은 방식입니다. VM이 죽어도 백업이 살아 있어야 합니다.

### 2. 워치독은 timeout이 기본값이어야 한다

헬스체크 명령은 실패만 고려하면 안 됩니다. 이번처럼 runtime이 굶주리면 `docker ps`는 실패하지 않고 그냥 멈춥니다. 그래서 워치독에는 `timeout`과 중복 실행 방지가 반드시 필요합니다.

### 3. vCPU 수는 장애 증폭 계수다

vCPU를 많이 주면 평소에는 좋지만, seccomp JIT나 RCU expedited path처럼 모든 CPU를 동기화하는 경로에서는 장애 증폭 계수가 됩니다. homelab runtime에는 "많을수록 좋다"보다 "멈춰도 호스트를 죽이지 않는 정도"가 더 중요합니다.

## 교훈 (Lessons Learned)

1. **VM reset 전에 디스크 이미지를 통째로 보존하자.** 라이브 추출이나 btrfs offline mount를 억지로 시도하기보다, 정지된 VM의 sparse raw 이미지를 먼저 보존하는 쪽이 훨씬 안전했다.
2. **로컬 PV는 편하지만 장애 경계가 좁다.** runtime, DB, 백업이 같은 VM 안에 있으면 "runtime 복구"와 "데이터 보존"이 서로 충돌한다.
3. **status가 resolved인지 monitoring인지 엄격하게 보자.** containerd storm은 제거됐지만 K8s 서비스와 PVC 데이터가 아직 fresh VM에 복원된 것은 아니다. 그래서 이 글은 `resolved`가 아니라 `monitoring`으로 둔다.
4. **워치독은 실패보다 hang을 먼저 가정해야 한다.** 장애 상황의 명령은 exit code를 주지 않을 수 있다. `timeout` 없는 health check는 운영 자동화가 아니라 새 장애 원인이 된다.
