---
title: "aily: AI 에이전트 세션을 폰에서 관리하기까지"
description: "금요일 밤 5줄짜리 bash 훅에서 시작해, 22일 만에 양방향 세션 브릿지가 되기까지. Claude Code, Codex, Gemini 에이전트의 알림을 Discord/Slack으로 받고, 폰에서 답장까지 보내는 도구를 만든 여정입니다."
date: 2026-03-01
slug: /aily-ai-session-bridge
tags: [AI, Claude, Discord, Slack, tmux, CLI, OpenSource, DevOps]
published: false
---

# aily: AI 에이전트 세션을 폰에서 관리하기까지

## 문제: "끝났나?"

AI 코딩 에이전트를 쓰다 보면, 자연스럽게 이런 워크플로우가 됩니다:

1. tmux 세션에서 Claude Code에 작업을 시킴
2. 시간이 걸리니까 자리를 비움
3. 돌아와서 터미널을 확인

문제는 3번입니다. 에이전트가 5분 전에 끝났을 수도 있고, 질문을 하고 기다리고 있을 수도 있습니다. SSH 호스트가 여러 대이고 세션이 동시에 돌아가면, **어떤 세션이 끝났는지 알 방법이 없습니다**.

> "그냥 Discord에 알림 하나 오면 안 되나?"

이 한 줄짜리 생각이 aily의 시작이었습니다.

## 금요일 밤: 5개 파일, 297줄

2월 7일 금요일 밤, 첫 커밋에는 파일이 5개뿐이었습니다.

```
hooks/
├── notify-clawdia.sh       # Claude Code 알림 훅
└── extract-last-message.py  # JSONL 파서
install.sh
.env.example
.gitignore
```

핵심 로직은 단순했습니다:

```bash
# Claude Code의 Notification 훅이 실행되면:
# 1. 5초 대기 (JSONL 파일에 응답이 쓰일 때까지)
# 2. extract-last-message.py로 마지막 어시스턴트 메시지 추출
# 3. Discord 스레드에 포스트
(
  sleep 5
  message=$(python3 extract-last-message.py)
  curl -X POST "discord.com/api/.../messages" -d "$message"
) &
disown
exit 0  # 훅은 즉시 리턴 (Claude Code 타임아웃 방지)
```

스레드 이름은 `[agent] <tmux 세션명>`. 세션마다 전용 스레드가 생기고, 에이전트의 응답이 거기에 올라옵니다.

"Clawdia"라는 이름은 Claude의 별명이었습니다 — 아직 이 도구가 Claude 전용이라고 생각했거든요.

## 첫 번째 전환점: AI 봇 → 결정론적 브릿지

원래 계획은 **Discord 봇이 AI로 메시지를 이해하고 전달**하는 것이었습니다. Clawdia 봇이 스레드의 메시지를 읽고, AI가 판단해서 tmux로 포워딩하는 구조.

하루 만에 포기했습니다.

AI가 중간에 끼면 **100% 신뢰할 수 없습니다**. 어떤 때는 정확히 전달하고, 어떤 때는 챗봇처럼 자체 응답을 합니다. 메시지 릴레이에 확률적 요소가 있으면 안 됩니다.

```python
# agent-bridge.py — AI 없는 결정론적 브릿지
# 규칙: [agent] 스레드의 메시지 → 해당 tmux 세션으로 전달. 끝.
async def on_message(message):
    if not message.thread.name.startswith("[agent] "):
        return
    session = message.thread.name.removeprefix("[agent] ")
    host = find_host_with_session(session)
    await ssh_send_keys(host, session, message.content)
```

이 결정이 프로젝트의 정체성을 바꿨습니다. **챗봇이 아니라 릴레이**. AI가 만든 도구지만, 동작 자체에 AI는 없습니다.

같은 날 이름도 바꿨습니다. `claude-hooks` → `aily`. Claude 전용이 아니게 됐으니까요.

## 양방향 통신

단방향 알림만으로는 부족했습니다. 에이전트가 "어떤 방법을 사용할까요?" 같은 질문을 하면, 폰에서 바로 답하고 싶었습니다.

```mermaid
flowchart LR
    A["Agent\n(Claude/Codex/Gemini)"] --> B["Hook\n(post.sh)"]
    B --> C["Discord 스레드"]
    B --> D["Slack 스레드"]
    C --> F["Bridge"]
    D --> F
    F -->|"SSH + tmux send-keys"| A
```

여기서 하나 재미있는 삽질이 있었습니다. `tmux send-keys`로 메시지를 보낼 때, 텍스트와 Enter를 한 번에 보내면 Claude Code가 Enter를 줄바꿈(Shift+Enter)으로 해석합니다.

```bash
# 이렇게 하면 안 됨 (Enter가 줄바꿈이 됨)
tmux send-keys -t session "message" Enter

# 두 단계로 나눠야 함
tmux send-keys -t session "message"
sleep 0.3
tmux send-keys -t session Enter
```

이런 건 문서에도 없습니다. 삽질해야 알 수 있는 것들.

## 멀티 플랫폼 디스패처

Discord만 지원하다가 Slack도 추가하면서, 아키텍처를 다시 생각했습니다.

```
notify-claude.sh  ─┐
notify-codex.py   ─┤──▶ post.sh (디스패처) ──┬──▶ discord-post.sh
notify-gemini.sh  ─┘                         └──▶ slack-post.sh
```

`post.sh`는 설정된 토큰을 보고 플랫폼을 자동 감지합니다. Discord 토큰만 있으면 Discord로, 둘 다 있으면 병렬로 전송. 에이전트별 훅은 플랫폼을 몰라도 됩니다.

tmux 세션 라이프사이클도 연동했습니다:

| 이벤트 | 동작 |
|--------|------|
| tmux 세션 생성 | Discord/Slack에 스레드 자동 생성 |
| 에이전트 작업 완료 | 스레드에 응답 포스트 |
| 에이전트가 질문 | 스레드에 선택지 포스트 |
| 사용자가 스레드에 답장 | tmux 세션으로 입력 전달 |
| tmux 세션 종료 | 스레드 아카이브 |

세션을 시작하면 스레드가 생기고, 세션을 죽이면 스레드가 아카이브됩니다. 수동 설정이 필요 없습니다.

## 대시보드: 3개의 AI가 함께 설계

> "웹에서도 세션 상태를 보고 싶다."

대시보드 기획을 **Claude, Codex, Gemini 3개의 에이전트가 동시에** 진행했습니다:

- **Claude** → 백엔드 아키텍처 설계 (700줄 아키텍처 문서)
- **Gemini** → UX/UI 디자인 스펙 (1,564줄 UI 명세)
- **Codex** → 기술 구현 명세 (2,259줄 코드 예시)

기술 스택은 의도적으로 가볍게 잡았습니다:

| 선택 | 이유 |
|------|------|
| SQLite (not PostgreSQL) | 1인용 도구. 동시 쓰기 경합 없음. 백업은 파일 복사 |
| aiohttp (not FastAPI) | 이미 브릿지에서 사용 중. 의존성 추가 없음 |
| Alpine.js (not React) | 빌드 파이프라인 없음. HTML 한 파일로 동작 |

과도한 엔지니어링을 피하려고 의식적으로 노력했습니다. 이 도구의 사용자는 한 명(나)이고, 읽기 위주의 워크로드입니다.

## CLI: 4번의 프롬프트로 끝나는 설정

초기 설정은 `.env` 파일을 수동으로 편집하는 방식이었습니다. 대시보드 URL, 인증 토큰, Discord 토큰, 채널 ID... 10개가 넘는 변수를 직접 입력해야 했습니다.

실제로 다른 기기에서 설치해보니 **너무 번거로웠습니다**. 특히 개인 사용자에게 대시보드는 필수가 아닌데, 첫 단계부터 대시보드 URL을 물어봤습니다.

설정 흐름을 완전히 재설계했습니다:

```
$ aily init

=== aily setup wizard ===

  1) Notification platform
     > discord / slack / both

  Discord bot token: ****
  Discord channel ID: 12345...
  ✓ Discord: ai-notifications

  Defaults: SSH=localhost, cleanup=archive, no dashboard
  Use defaults? [Y/n]: y

  ✓ Saved to ~/.config/aily/env
  ✓ Hooks installed

=== Setup complete ===
```

**4번의 입력으로 끝납니다.** 플랫폼 선택 → 토큰 → 채널 ID → "기본값 쓸래?" → 완료.

대시보드, SSH 호스트, 에이전트 자동 실행 같은 고급 설정은 "기본값 쓸래?"에서 `n`을 눌렀을 때만 나옵니다. localhost 대시보드를 선택하면 인증 토큰도 자동으로 생성합니다.

## 설정 경로 마이그레이션

처음에 설정 파일이 `~/.claude/hooks/.notify-env`에 있었습니다. Claude Code 훅 디렉토리 안에요. 이건 aily가 Claude Code 전용 훅이던 시절의 잔재였습니다.

이제 Claude, Codex, Gemini, OpenCode를 모두 지원하는데, 설정이 `.claude/` 안에 있는 건 맞지 않습니다.

[XDG Base Directory 스펙](https://specifications.freedesktop.org/basedir-spec/latest/)을 따라 `~/.config/aily/env`로 이전했습니다. 모든 훅, 브릿지, CLI가 새 경로를 먼저 확인하고, 없으면 이전 경로로 폴백합니다. `aily init`을 다시 실행하면 자동으로 마이그레이션됩니다.

## 실사용에서 발견한 것들

직접 쓰면서 발견한 문제들이 가장 중요한 개선으로 이어졌습니다:

### tmux 세션 감지 버그

`tmux display-message -p '#S'`가 **훅이 실행되는 세션이 아니라 attach된 클라이언트의 세션 이름을 반환**했습니다. 세션 A에서 실행된 훅이 세션 B의 이름을 리포트하는 상황.

```bash
# 잘못된 방법 (attach된 클라이언트의 세션)
TMUX_SESSION=$(tmux display-message -p '#S')

# 올바른 방법 (현재 pane의 세션)
TMUX_SESSION=$(tmux display-message -t "${TMUX_PANE}" -p '#{session_name}')
```

`$TMUX_PANE` 환경변수를 사용해야 합니다. 이건 tmux가 모든 pane에 설정하는 변수인데, 대부분의 tmux 관련 글에서 언급하지 않습니다.

### 토큰 붙여넣기가 안 보이는 문제

`read -rsp`로 시크릿을 입력받으면 `-s` 플래그 때문에 **붙여넣기를 해도 아무것도 표시되지 않습니다**. 사용자는 붙여넣기가 된 건지 안 된 건지 알 수 없습니다.

입력 후 `****`를 표시하도록 수정했습니다. 작은 변경이지만 UX 차이가 큽니다.

### Rust 재작성은 불필요

성능 개선을 위해 Rust로 재작성하는 것을 검토했습니다. 프로파일링 결과:

- SSH 네트워크 I/O: **79%**
- 플랫폼 API 호출: **21%**
- CPU (파싱, 로직): **0.1%**

Rust로 바꿔봤자 0.1%가 빨라질 뿐입니다. 대신 SSH ControlMaster 연결 재사용, 병렬 호스트 스캔, HTTP 세션 재사용으로 체감 성능을 크게 개선했습니다.

## AI로 AI 도구를 만든다는 것

aily 자체가 메타적인 프로젝트입니다. **AI 에이전트 세션을 관리하는 도구를, AI 에이전트 세션으로 만들었습니다.**

22일 동안의 개발 과정:

- **8개의 Claude Code 세션** (가장 긴 세션은 5일간 연속 대화)
- **132개의 커밋**
- **멀티 에이전트 기획**: Claude, Gemini, Codex가 동시에 설계 문서 작성
- **백그라운드 구현**: git worktree에서 각각의 에이전트가 병렬로 코드 작성
- **리뷰 루프**: Gemini code assist로 5라운드 보안/동시성 리뷰

아이러니하게도, aily를 개발하는 동안 aily가 가장 필요했습니다. "Claude가 끝났나?" 확인하려고 터미널을 왔다갔다 하면서, 바로 그 문제를 해결하는 도구를 만들고 있었거든요.

## 현재 구조

297줄에서 시작한 프로젝트의 현재 모습:

```
aily/
├── hooks/              # 에이전트별 알림 훅 (bash/python)
│   ├── post.sh         # 멀티 플랫폼 디스패처
│   ├── notify-claude.sh
│   ├── notify-codex.py
│   └── notify-gemini.sh
├── agent-bridge.py     # Discord ↔ tmux 양방향 브릿지
├── slack-bridge.py     # Slack ↔ tmux 양방향 브릿지
├── dashboard/          # 웹 대시보드 (aiohttp + Alpine.js)
├── aily                # CLI (setup, status, doctor, sessions)
├── Dockerfile          # 멀티 모드 컨테이너
└── install.sh          # 원클릭 설치
```

| 기능 | 설명 |
|------|------|
| 에이전트 알림 | Claude, Codex, Gemini, OpenCode 작업 완료 시 알림 |
| 양방향 채팅 | Discord/Slack 스레드에서 답장 → 에이전트에 입력 전달 |
| 세션 라이프사이클 | tmux 세션 생성/종료 시 스레드 자동 관리 |
| 멀티 호스트 | SSH로 여러 대의 개발 머신 관리 |
| 대시보드 | 실시간 세션 모니터링, 메시지 히스토리 |
| 사용량 모니터링 | API 사용량 추적, 리밋 리셋 시 자동 실행 |

## 설치

```bash
git clone https://github.com/jiunbae/aily.git
cd aily && ./aily init
```

4번의 입력이면 됩니다. 소스는 [GitHub](https://github.com/jiunbae/aily)에서 확인할 수 있습니다.

---

**TL;DR**: 금요일 밤 "에이전트 끝나면 알림 좀 받자"로 시작한 프로젝트가, 22일 뒤에는 멀티 에이전트, 멀티 플랫폼, 양방향 세션 브릿지가 됐습니다. 직접 쓰면서 만들었기 때문에, 모든 기능이 실제 필요에서 나왔습니다.
