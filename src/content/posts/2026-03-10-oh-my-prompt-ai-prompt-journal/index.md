---
title: "매일 300개의 프롬프트를 쓰면서 아무것도 기억하지 못했던 이야기"
description: "AI 코딩 에이전트에게 보내는 프롬프트를 자동으로 캡처하고 분석하는 CLI 도구 Oh My Prompt를 만들게 된 배경과 기술적 여정을 공유합니다."
date: 2026-03-10
permalink: /oh-my-prompt-ai-prompt-journal
tags: [AI, CLI, Claude, Codex, OpenCode, Gemini, Productivity, OpenSource]
published: true
---

# 매일 300개의 프롬프트를 쓰면서 아무것도 기억하지 못했던 이야기

## 들어가며

어느 날 문득 이런 생각이 들었습니다.

> "나는 오늘 AI에게 뭐라고 말했지?"

Claude Code, Codex, OpenCode, Gemini CLI — 요즘 하루 종일 AI 코딩 에이전트와 대화하면서 개발합니다. "이 함수 리팩터링해줘", "테스트 추가해줘", "이 버그 원인 찾아줘". 체감상 하루에 수백 개의 프롬프트를 작성하고 있었죠.

그런데 세션이 끝나면 그 대화는 그냥 사라집니다. 어제 어떤 프롬프트로 좋은 결과를 얻었는지, 어떤 프로젝트에서 AI를 가장 많이 활용했는지, 비슷한 요청을 몇 번이나 반복하고 있는지 — 아무것도 알 수 없었어요.

개발자에게 Git 히스토리가 있고 APM 모니터링이 있듯이, AI와의 대화에도 기록과 분석이 필요하지 않을까? 이 질문에서 [Oh My Prompt](https://github.com/jiunbae/oh-my-prompt)가 시작되었습니다.

## 우리가 잃어버리고 있는 것

잠시 상황을 정리해볼게요.

**개발자 A**는 Claude Code를 매일 사용합니다. 프롬프트를 정성껏 쓰고, 결과가 좋으면 "다음에도 이렇게 써야지" 하고 넘어갑니다. 하지만 다음 날이면 그 프롬프트를 잊어버리죠. 같은 시행착오를 반복합니다.

**개발자 B**도 매일 사용합니다. 어떤 프로젝트에서 얼마나 AI를 활용했는지, 어떤 패턴의 프롬프트가 효과적이었는지 데이터로 볼 수 있습니다. 자연스럽게 프롬프트 작성 습관이 개선됩니다.

차이는 단순합니다. **기록하느냐, 안 하느냐.**

문제는 이 기록을 수동으로 하기엔 현실적으로 불가능하다는 점이에요. 프롬프트를 쓸 때마다 어딘가에 복사해놓는 건 개발 흐름을 깨뜨리죠. 그래서 자동으로, 투명하게, 기존 워크플로우를 전혀 방해하지 않으면서 모든 프롬프트를 캡처하는 도구가 필요했습니다.

## 캡처 전략: 훅 시스템 활용하기

프롬프트를 자동으로 캡처하려면 AI 에이전트와 개발자 사이에 끼어들어야 합니다. 몇 가지 방법을 검토했어요.

| 방식 | 장점 | 단점 |
|------|------|------|
| 브라우저 확장 | 웹 UI 지원 | CLI 에이전트는 못 잡음 |
| 프록시 서버 | 모든 트래픽 캡처 | HTTPS 인증서 문제, 과도한 복잡성 |
| **에이전트 훅** | **투명, 공식 지원** | **에이전트마다 형식이 다름** |

결론적으로 각 에이전트가 공식으로 제공하는 훅 시스템을 활용하기로 했습니다. Claude Code는 `.claude/hooks/`에 쉘 스크립트를 둘 수 있고, Codex는 `codex.json`의 `notify` 필드를, OpenCode는 `opencode.json`의 `plugin` 배열을, Gemini CLI는 커스텀 확장 훅을 사용합니다. 네 개의 에이전트 모두 `omp setup`에서 자동 감지되고 훅이 설치됩니다.

데이터 흐름은 이렇게 됩니다:

```
에이전트 사용 → 훅 트리거 → omp ingest → SQLite 저장
                                            │
                                    omp sync → 서버 업로드
                                                 │
                                             대시보드에서 분석
```

훅이 한 번 설치되면 사용자는 아무것도 신경 쓸 필요가 없어요. 평소처럼 `claude "이 함수 리팩터링해줘"`라고 입력하면, 그 프롬프트는 자동으로 로컬 SQLite에 기록됩니다.

## 셋업 경험을 다시 생각하다

도구를 만들었으니 사람들이 써야 하는데, 첫 셋업 경험이 형편없었습니다.

처음에는 Node.js `readline`으로 구현했어요. 동작은 하지만 밋밋했죠:

```bash
# Before — readline 기반
$ omp setup

  Oh My Prompt - Setup Wizard
  ============================

  [1/4] Server URL
  > Server URL [https://prompt.jiun.dev]:
  > Choice [1]:
  > Email:
  > Password:
  Authenticating... OK

  Setup complete!
```

보이시나요? 아무런 시각적 피드백이 없습니다. 스피너도 없고, 색상도 없고, 어디까지 진행되었는지 감도 잡히지 않습니다. 2026년의 CLI라고 하기엔 좀 부끄러웠어요.

[Astro CLI](https://astro.build)나 [Create T3 App](https://create.t3.gg)의 셋업 경험을 보면, 예쁜 프롬프트와 스피너가 "이 도구는 잘 만들어져 있구나"라는 첫인상을 줍니다. 개발자 도구에서 CLI UX는 생각보다 중요하죠.

그래서 [`@clack/prompts`](https://github.com/bombshell-dev/clack)를 도입했습니다. 결과는 이렇게 바뀌었어요:

```
# After — @clack/prompts 기반
$ omp setup

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
◇  Authenticating... Logged in as user@example.com
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
│  ◻ Gemini CLI (detected)
│
◇  Hooks installed (Claude Code, Codex, Gemini CLI)
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

단순히 라이브러리를 교체한 게 아니라 UX 설계가 달라졌습니다. `select`로 인증 방식을 고르고, `multiselect`로 설치할 훅을 선택하고, `spinner`로 비동기 작업의 진행 상황을 보여주고, `note` 박스로 최종 결과를 정리합니다.

## CJS 프로젝트에서 ESM-only 패키지 쓰기

여기서 한 가지 기술적 도전이 있었는데요. `@clack/prompts`는 ESM-only 패키지입니다. 그런데 Oh My Prompt CLI는 CJS예요. `better-sqlite3` 같은 네이티브 모듈과의 호환성 때문에 전체 프로젝트를 ESM으로 전환하기는 어려웠죠.

처음에는 "그냥 ESM으로 전환하자"고 생각했지만, 그러면 기존의 `require()` 기반 코드를 전부 바꿔야 하고, 네이티브 모듈 호환성도 검증해야 합니다. 리스크 대비 이득이 맞지 않았어요.

대신 동적 `import()`로 해결했습니다:

```javascript
// src/omp/ui.js — CJS 파일
const pc = require("picocolors");  // CJS → 동기 로드, 항상 사용 가능

let _clack = null;
async function loadClack() {
  if (!_clack) {
    _clack = await import("@clack/prompts");  // ESM → 필요할 때만 비동기 로드
  }
  return _clack;
}

module.exports = { c: pc, loadClack, /* ... */ };
```

이 패턴의 핵심은 **두 세계를 분리한 것**입니다:

- `picocolors`(2KB, CJS)는 `require()`로 즉시 로드 — 컬러 출력은 어디서든 동기적으로 사용 가능
- `@clack/prompts`(ESM)는 `await import()`로 지연 로드 — 인터랙티브 프롬프트가 실제로 필요한 순간에만 로드

비대화형 모드(`--yes`)에서는 `loadClack()`이 호출되지 않으므로 ESM 패키지가 아예 로드되지 않습니다. CI 환경에서 불필요한 오버헤드가 없죠.

## 데이터 저장: SQLite + PostgreSQL 이중 구조

데이터 저장소 설계에서도 고민이 있었습니다.

클라이언트(CLI)에서 PostgreSQL을 직접 쓸 수도 있었지만, 그러면 사용자에게 DB 서버를 요구하게 됩니다. 로컬에서 편하게 쓰려고 만든 CLI인데, 외부 의존성이 생기면 본말전도죠.

그래서 클라이언트는 SQLite, 서버는 PostgreSQL을 쓰는 이중 구조로 설계했습니다:

```
[CLI]                                    [Server]
SQLite (omp.db)                         PostgreSQL
  │                                        │
  ├─ 프롬프트 즉시 저장                     ├─ 풀텍스트 검색
  ├─ 오프라인 동작                          ├─ 다중 사용자
  ├─ 파일 하나로 관리                       ├─ 분석 쿼리
  └─ 설치 불필요                           └─ 대시보드 서빙
            │
            └── omp sync ──── 청크 업로드 (500건 단위)
                              content_hash로 중복 감지
```

SQLite는 `npm install` 할 때 함께 설치되는 `better-sqlite3`로 관리합니다. 별도의 DB 서버가 필요 없고, `~/.config/oh-my-prompt/omp.db` 파일 하나에 모든 데이터가 담깁니다. 비행기 안에서도 프롬프트는 계속 캡처되고, 나중에 와이파이가 되면 `omp sync` 한 번이면 서버에 올라갑니다.

동기화는 청크 단위(기본 500건)로 POST 요청을 보냅니다. 각 레코드의 `content_hash`로 중복을 감지하기 때문에, 같은 데이터를 여러 번 동기화해도 안전합니다. 네트워크가 중간에 끊겨도 다음 동기화에서 이어서 전송합니다.

## 기존 프롬프트 살리기: backfill

이미 AI 코딩 에이전트를 오래 사용해왔다면, 축적된 대화 기록이 있을 겁니다. Oh My Prompt를 지금 설치했다고 과거의 데이터를 포기할 필요는 없어요.

```bash
omp backfill                    # 4개 에이전트 모두
omp backfill --claude-only      # Claude 트랜스크립트만
omp backfill --codex-only       # Codex 히스토리만
omp backfill --opencode-only    # OpenCode 세션만
omp backfill --gemini-only      # Gemini 채팅만
omp backfill --dry-run           # 미리보기 (실제 저장 안 함)
```

각 에이전트의 데이터 형식이 다르기 때문에, 백필 구현은 꽤 까다로웠습니다:

| 에이전트 | 데이터 위치 | 형식 |
|----------|-------------|------|
| Claude Code | `~/.claude/projects/` | 세션별 JSONL |
| Codex | `~/.codex/history.jsonl` + `~/.codex/sessions/` | 단일 JSONL + 세션 JSONL |
| OpenCode | `~/.local/share/opencode/opencode.db` | SQLite (session/message/part 테이블) |
| Gemini CLI | `~/.gemini/tmp/*/chats/session-*.json` | 프로젝트 해시별 JSON |

특히 Gemini는 메시지 `content`가 문자열일 때도 있고 `[{text: "..."}]` 배열일 때도 있어서, 두 형식 모두 처리하는 로직이 필요했습니다. OpenCode는 SQLite DB를 직접 열어서 세션 → 메시지 → 파트 순으로 조인하는 방식이라, better-sqlite3를 이미 쓰고 있던 게 다행이었죠.

백필 후 서버에 동기화할 때도 주의할 점이 있었습니다. 백필된 레코드는 원래 생성 시점의 타임스탬프를 갖기 때문에, 동기화 커서(마지막으로 동기화한 시점)보다 과거에 위치합니다. 이걸 해결하기 위해 백필 후 새 레코드가 import되면 동기화 커서를 자동으로 리셋하도록 구현했습니다. 사용자가 `--since`를 직접 지정할 필요 없이, `omp backfill` → `omp sync`만 실행하면 됩니다.

중복 감지 로직(`content_hash` + `event_id`)이 있어서 `backfill`을 여러 번 실행해도 같은 프롬프트가 두 번 저장되지 않습니다. 안심하고 돌려도 됩니다.

## `omp doctor`로 상태 한눈에 보기

셋업이 잘 되었는지, 훅이 제대로 설치되었는지, 서버 연결은 정상인지 — 이런 것들을 일일이 확인하는 건 번거롭습니다. `omp doctor`는 모든 상태를 한 번에 확인해줍니다.

```bash
$ omp doctor
✔ Doctor: all checks passed

# 문제가 있을 때는 구체적으로 알려줍니다:
$ omp doctor
✖ server.url is set but server.token is missing
▲ Auto-sync is enabled but daemon is not running
▲ queue has pending items; run 'omp ingest --replay'
```

DB 상태, 훅 설치 여부, 큐에 쌓인 미처리 항목, 동기화 상태, 자동 동기화 데몬 상태까지 확인합니다. `picocolors`로 `✔`(초록), `✖`(빨강), `▲`(노랑) 아이콘을 써서 한눈에 상태가 보이죠.

## Docker 없이도 대시보드를: SQLite 로컬 모드

서버 배포와 Docker 기반 로컬 모드를 만들었지만, 한 가지 질문이 계속 남아있었습니다.

> "Docker도 없는 일반 사용자는 어떻게 하지?"

npm으로 CLI를 설치한 사람이 대시보드를 보려면 Docker를 따로 설치해야 한다는 건, 진입 장벽이 너무 높았습니다. 그래서 **Node.js 내장 `http` 모듈만으로** 돌아가는 SQLite 로컬 대시보드를 만들었습니다. 의존성 추가 없이, CLI가 이미 쓰고 있는 SQLite DB를 그대로 읽어서 웹 UI를 제공합니다.

```bash
omp serve --local                  # 즉시 http://localhost:3000
omp serve --local --port 8080      # 포트 변경
omp serve --local --host 0.0.0.0   # 네트워크 노출
```

기술적으로는 꽤 제약이 많은 환경이었습니다. React도 없고, 번들러도 없고, 템플릿 엔진도 없습니다. HTML 문자열을 직접 조립해서 `res.end()`로 내보내는 방식이죠. 그래도 대시보드, 프롬프트 목록, 세션 상세, 풀텍스트 검색(FTS5)까지 지원합니다. 다크 테마도 CSS만으로 구현했습니다.

이 모드의 핵심은 **동기화가 필요 없다**는 점입니다. CLI가 캡처한 프롬프트는 이미 로컬 SQLite에 있고, 대시보드는 그 DB를 직접 쿼리합니다. `omp serve --local` 한 줄이면 끝.

결과적으로 Oh My Prompt는 세 가지 모드를 지원하게 되었습니다:

| 모드 | 필요한 것 | 대상 |
|------|-----------|------|
| SQLite 로컬 | 없음 | 가장 빠른 시작, 개인 사용 |
| Docker 로컬 | Docker | 풀 기능, 개인 사용 |
| 서버 배포 | Docker + 도메인 | 팀, 멀티 디바이스 |

## 정리하며

Oh My Prompt가 해결하려는 문제를 다시 정리해볼게요:

| 문제 | 해결 |
|------|------|
| AI와의 대화가 세션 종료 후 사라짐 | 훅으로 자동 캡처, SQLite에 영속 저장 |
| 기존 대화 기록을 활용할 수 없음 | `backfill`로 4개 에이전트 과거 기록 임포트 |
| 멀티 디바이스에서 데이터가 분산됨 | `sync`로 서버에 통합 |
| 프롬프트 패턴을 분석할 방법이 없음 | 셀프 호스팅 대시보드 (Docker 없이도 가능) |
| CLI 도구의 UX가 투박함 | `@clack/prompts`로 현대적 인터랙티브 UI |

```bash
npm install -g oh-my-prompt
omp setup                   # 에이전트 자동 감지, 훅 설치
omp backfill                # Claude, Codex, OpenCode, Gemini 히스토리 임포트
omp serve --local           # 로컬 대시보드 즉시 실행
```

네 줄이면 시작할 수 있습니다.

## 그럼에도 불구하고

솔직히 말씀드리면, 아직 갈 길이 멀어요.

현재는 프롬프트를 **저장하고 동기화하는 것**까지만 잘 합니다. 정작 "어떤 프롬프트가 좋은 프롬프트인가"를 판단하는 기능은 아직 없어요. 프롬프트 품질 스코어링 — 구조, 구체성, 컨텍스트 제공 여부를 분석해서 점수를 매기는 기능 — 은 가장 만들고 싶은 기능이지만, "좋은 프롬프트"의 기준을 정의하는 것 자체가 연구 과제입니다.

주간/월간 자동 리포트로 "이번 주에 가장 많이 쓴 프롬프트 패턴", "프로젝트별 AI 의존도 변화" 같은 인사이트를 주고 싶은데, 어떤 지표가 의미 있는지 아직 실험 중이에요. 데이터는 쌓이고 있으니, 어떻게 활용할지는 사용하면서 발견해가려 합니다.

팀 단위 기능도 빠져 있습니다. "우리 팀에서 가장 효과적인 코드 리뷰 프롬프트는 뭘까?" 같은 질문에 답할 수 있으면 좋겠지만, 아직은 개인 도구 단계에 머물러 있죠.

그래도 확신하는 건 있습니다. AI 에이전트를 더 잘 활용하려면 **"내가 AI에게 어떻게 말하고 있는지"를 먼저 알아야 한다**는 것. 측정할 수 없는 것은 개선할 수 없으니까요. Oh My Prompt는 그 측정의 첫 걸음입니다.

---

**프로젝트**: [jiunbae/oh-my-prompt](https://github.com/jiunbae/oh-my-prompt) | [npm](https://www.npmjs.com/package/oh-my-prompt)
