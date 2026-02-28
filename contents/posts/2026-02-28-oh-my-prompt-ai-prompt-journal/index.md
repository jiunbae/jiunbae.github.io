---
title: "Oh My Prompt: AI 코딩 에이전트를 위한 프롬프트 저널 만들기"
description: "매일 수백 개의 프롬프트를 쓰면서도 어떤 프롬프트가 효과적인지 모르고 있었습니다. 프롬프트를 자동으로 캡처하고, 분석하고, 시각화하는 CLI + 대시보드를 만든 이야기입니다."
date: 2026-02-28
slug: /oh-my-prompt-ai-prompt-journal
tags: [AI, CLI, Claude, Productivity, OpenSource]
published: true
---

# Oh My Prompt: AI 코딩 에이전트를 위한 프롬프트 저널 만들기

Claude Code, Codex, OpenCode 같은 AI 코딩 에이전트를 하루 종일 사용하다 보면, 하루에 수백 개의 프롬프트를 작성하게 됩니다. 그런데 문득 이런 생각이 들었습니다.

> "나는 어떤 프롬프트를 잘 쓰고, 어떤 프롬프트를 못 쓰고 있을까?"

세션이 끝나면 대화 내용은 흩어지고, 어떤 프로젝트에서 얼마나 AI를 활용했는지, 어떤 패턴의 프롬프트가 좋은 결과를 냈는지 알 수가 없었습니다. 이 문제를 해결하기 위해 **Oh My Prompt**를 만들었습니다.

## Oh My Prompt가 하는 일

한 문장으로 요약하면: **AI 에이전트에게 보내는 모든 프롬프트를 자동으로 캡처해서 로컬 DB에 저장하고, 대시보드로 분석할 수 있게 해주는 CLI 도구**입니다.

```
  You                    CLI                      Dashboard
  ───                    ───                      ─────────

  claude "fix the bug"
       │
       └──── hook ────▶  omp ingest ──▶ SQLite (local)
                              │
                              ├── omp sync ──▶ Server
                              │                  │
                              │              Dashboard
                              │              (self-hosted)
```

핵심 설계 원칙은 다음과 같습니다:

- **투명한 캡처**: 쉘 훅이 백그라운드에서 동작하므로 기존 워크플로우를 방해하지 않습니다
- **로컬 우선**: 모든 데이터는 SQLite에 먼저 저장됩니다. 오프라인에서도 동작합니다
- **셀프 호스팅**: 서버와 대시보드는 본인의 인프라에서 운영합니다. 데이터가 외부로 나가지 않습니다

## 지원하는 AI 에이전트

현재 세 가지 AI 코딩 에이전트를 지원합니다:

| 에이전트 | 훅 방식 | 자동 감지 |
|---------|---------|----------|
| [Claude Code](https://code.claude.com) | `.claude/hooks/` PreToolUse 훅 | `claude` 커맨드 또는 `~/.claude/` 디렉토리 |
| [OpenAI Codex](https://openai.com/codex) | `codex.json` notify 설정 | `codex` 커맨드 또는 `~/.codex/` 디렉토리 |
| [OpenCode](https://opencode.ai) | `opencode.json` plugin 설정 | `opencode` 커맨드 또는 `~/.config/opencode/` 디렉토리 |

`omp setup`을 실행하면 설치된 에이전트를 자동으로 감지하고 훅을 설치합니다. 훅이 설치되면 평소처럼 에이전트를 사용하기만 하면 프롬프트가 자동으로 캡처됩니다.

## CLI: 현대적인 인터랙티브 UX

최근에 CLI의 UX를 대폭 개선했습니다. [@clack/prompts](https://github.com/bombshell-dev/clack)와 [picocolors](https://github.com/alexeyraspopov/picocolors)를 도입해서 Astro CLI나 Create T3 App 같은 느낌의 인터랙티브 프롬프트를 구현했습니다.

### 설치

```bash
npm install -g oh-my-prompt
```

### 셋업 위저드

`omp setup`을 실행하면 스피너, 컬러, 선택형 프롬프트가 있는 대화형 위저드가 시작됩니다:

```
┌  oh-my-prompt
│
◆  Server URL
│  https://prompt.jiun.dev
│
◆  Authentication
│  ● Login with email & password (recommended)
│  ○ Paste existing API token
│
◆  Email
│  user@example.com
│
◆  Password
│  ********
│
◇  Authenticating... done
│
◆  Device name
│  my-macbook
│
◇  Database migrated (schema v3)
│
◆  Install hooks
│  ◻ Claude Code (detected)
│  ◻ Codex (detected)
│  ◻ OpenCode (not found)
│
◇  Hooks installed (Claude Code, Codex)
│
◇  Server validated (200)
│
◇  Setup Complete ───────────────────╮
│                                    │
│  Server:  https://prompt.jiun.dev  │
│  Device:  my-macbook               │
│  Hooks:   claude, codex            │
│                                    │
├────────────────────────────────────╯
│
└  Run omp backfill to import existing prompts
```

readline 기반의 밋밋한 프롬프트에서 `@clack/prompts` 기반의 인터랙티브 UI로 바꾸면서 고려한 점이 몇 가지 있습니다:

- **CJS 호환성 유지**: `@clack/prompts`는 ESM-only 패키지인데, 프로젝트가 CJS입니다. `await import()`로 동적 임포트해서 해결했습니다
- **비대화형 모드 보존**: `--yes` 플래그나 CI 환경에서는 clack을 로드하지 않고 기존 동작을 유지합니다
- **Ctrl-C 처리**: 모든 프롬프트에서 `isCancel()` 체크를 걸어서 깔끔하게 종료합니다

### 컬러 출력

`omp status`나 `omp doctor` 같은 명령도 컬러 출력을 지원합니다:

```bash
$ omp doctor
✔ Doctor: all checks passed

# 문제가 있을 때:
$ omp doctor
✖ server.url is set but server.token is missing

▲ Auto-sync is enabled but daemon is not running
```

## 아키텍처

Oh My Prompt는 크게 세 부분으로 구성됩니다:

### 1. CLI (`src/omp/`)

Node.js로 작성된 CLI 도구입니다. SQLite를 로컬 스토리지로 사용하고, 서버와의 동기화를 담당합니다.

```
src/omp/
├── cli.js          # 메인 커맨드 라우터
├── setup.js        # 대화형 셋업 위저드
├── ui.js           # UI 추상화 (clack + picocolors)
├── hooks.js        # 에이전트별 훅 설치/제거
├── ingest.js       # 프롬프트 수집 및 큐 처리
├── sync.js         # 서버 동기화
├── backfill.js     # 기존 트랜스크립트 임포트
├── serve.js        # 로컬 대시보드 (Docker)
└── ...
```

핵심 의존성은 세 개뿐입니다:

| 패키지 | 용도 | 크기 |
|--------|------|------|
| `better-sqlite3` | 로컬 프롬프트 저장소 | Native |
| `@clack/prompts` | 인터랙티브 프롬프트 UI | ESM, lazy-load |
| `picocolors` | 터미널 컬러 출력 | 2KB, CJS |

### 2. 서버 (`src/app/`)

Next.js 기반의 웹 애플리케이션입니다. REST API로 동기화 엔드포인트를 제공하고, PostgreSQL에 데이터를 저장합니다.

### 3. 대시보드

셀프 호스팅 가능한 웹 대시보드입니다. 프롬프트 검색, 프로젝트별 분석, 활동 히트맵, 토큰 사용량 추적 등을 제공합니다.

## 주요 기능

### 기존 프롬프트 임포트

이미 Claude Code나 Codex를 사용하고 있었다면, 기존 대화 기록을 한 번에 가져올 수 있습니다:

```bash
omp backfill                    # Claude + Codex 모두
omp backfill --claude-only      # Claude Code만
omp backfill --codex-only       # Codex만
```

Claude Code의 경우 `~/.claude/projects/`에 있는 JSONL 트랜스크립트 파일을 파싱합니다. Codex의 경우 `~/.codex/history.jsonl` 파일을 읽습니다. 중복 감지가 내장되어 있어서 여러 번 실행해도 안전합니다.

### 서버 동기화

로컬에 쌓인 프롬프트를 서버로 업로드합니다:

```bash
omp sync              # 수동 동기화
omp sync auto         # 자동 동기화 데몬 시작
omp sync auto stop    # 데몬 중지
```

자동 동기화를 켜면 프롬프트가 캡처될 때마다 디바운스(30초)를 거쳐 자동으로 서버에 업로드됩니다. 오프라인 상태에서 캡처된 프롬프트도 다음 동기화 때 함께 올라갑니다.

### 로컬 대시보드

서버를 따로 배포하기 번거롭다면, Docker로 로컬에서 바로 실행할 수 있습니다:

```bash
omp serve              # PostgreSQL + Redis + App 시작
# Dashboard: http://localhost:3000
```

`omp serve` 명령 하나로 Docker Compose가 PostgreSQL, Redis, App 컨테이너를 자동으로 구성합니다. Docker 이미지 풀, 컨테이너 시작, 헬스체크까지 스피너로 진행 상황을 보여줍니다.

### 상태 확인

```bash
$ omp status
Server:  https://prompt.jiun.dev
Token:   abc12345...
Storage: sqlite
SQLite:  ~/.config/oh-my-prompt/omp.db
Capture response: on
Hooks:   claude=installed, codex=installed, opencode=not installed
Last capture: 2026-02-28T14:30:00Z
Queue:   0 files, 0 bytes
```

## 개발 과정에서의 선택들

### 왜 CLI인가

프롬프트 캡처를 웹 확장 프로그램이나 프록시로 구현할 수도 있었지만, CLI를 선택한 이유가 있습니다:

1. **쉘 훅이 가장 투명합니다**: AI 에이전트들이 제공하는 훅 시스템을 그대로 활용하면 에이전트 동작에 영향을 주지 않습니다
2. **오프라인 동작**: 네트워크에 의존하지 않고 로컬에 먼저 저장합니다
3. **이식성**: `npm install -g`로 어디서든 설치할 수 있습니다

### 왜 SQLite + PostgreSQL 이중 구조인가

클라이언트(CLI)는 SQLite를 쓰고, 서버는 PostgreSQL을 씁니다. 이유는 간단합니다:

- **SQLite**: 설치가 필요 없고, 파일 하나로 관리됩니다. CLI에 DB 서버를 요구하는 것은 과도합니다
- **PostgreSQL**: 다중 사용자, 풀텍스트 검색, 분석 쿼리에 적합합니다. 대시보드에는 이런 기능이 필요합니다

동기화할 때 SQLite의 데이터를 API로 PostgreSQL에 업로드하는 구조입니다. 청크 단위(기본 500건)로 나눠서 전송하고, 중복 감지는 `content_hash`로 처리합니다.

### ESM-only 패키지를 CJS에서 사용하기

`@clack/prompts`는 ESM-only 패키지인데, Oh My Prompt CLI는 CJS입니다. 전체 프로젝트를 ESM으로 전환하는 대신 동적 임포트로 해결했습니다:

```javascript
// src/omp/ui.js
const pc = require("picocolors");  // CJS — 항상 사용 가능

let _clack = null;
async function loadClack() {
  if (!_clack) {
    _clack = await import("@clack/prompts");  // ESM — 필요할 때만 로드
  }
  return _clack;
}
```

이 방식의 장점은:
- CJS 프로젝트를 유지하면서도 ESM 패키지를 사용할 수 있습니다
- `picocolors`는 CJS이므로 동기적으로 즉시 사용 가능합니다
- `@clack/prompts`는 인터랙티브 프롬프트가 필요한 순간에만 로드됩니다
- 비대화형 모드(`--yes`)에서는 clack을 아예 로드하지 않아서 불필요한 오버헤드가 없습니다

## 사용 시작하기

```bash
# 설치
npm install -g oh-my-prompt

# 셋업 (대화형 위저드)
omp setup

# 기존 프롬프트 임포트
omp backfill

# 서버로 동기화
omp sync

# 상태 확인
omp doctor
```

관심이 있으시다면 [GitHub 레포지토리](https://github.com/jiunbae/oh-my-prompt)를 확인해주세요. 이슈나 PR도 환영합니다.

## 앞으로의 계획

현재 구현된 기능은 캡처와 동기화에 집중되어 있습니다. 앞으로 추가하고 싶은 것들이 있습니다:

- **프롬프트 품질 스코어링**: 프롬프트의 구조, 구체성, 컨텍스트 제공 여부를 분석해서 점수를 매기는 기능
- **패턴 리포트**: 주간/월간 단위로 프롬프트 사용 패턴을 분석하는 자동 리포트
- **팀 기능**: 팀 단위로 프롬프트 패턴을 비교하고 베스트 프랙티스를 공유하는 기능

AI 에이전트를 더 잘 활용하기 위해서는 "내가 AI에게 어떻게 말하고 있는지"를 알아야 한다고 생각합니다. Oh My Prompt가 그 첫 걸음이 되었으면 합니다.

---

**프로젝트 링크**: [jiunbae/oh-my-prompt](https://github.com/jiunbae/oh-my-prompt) | [npm](https://www.npmjs.com/package/oh-my-prompt)
