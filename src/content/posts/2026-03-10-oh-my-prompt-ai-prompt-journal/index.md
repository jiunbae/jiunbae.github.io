---
title: "오늘 AI에게 뭐라고 말했더라: Oh My Prompt 5주 기록"
description: "Claude Code, Codex, OpenCode, Gemini CLI에 보낸 프롬프트를 자동으로 캡처하는 도구를 만든 5주. MinIO를 걷어낸 뻘짓, npm 버저닝 삽질, 하룻밤에 로컬 대시보드까지 밀어붙인 3월 9일의 기록."
date: 2026-03-10
permalink: /oh-my-prompt-ai-prompt-journal
tags: [AI, CLI, Claude, Codex, OpenCode, Gemini, Productivity, OpenSource]
published: true
---

# 오늘 AI에게 뭐라고 말했더라

요즘 하루 종일 AI 코딩 에이전트와 대화하면서 개발합니다. Claude Code, Codex, OpenCode, Gemini CLI를 번갈아 쓰는데, 세션이 끝나면 그 대화는 그냥 사라집니다. 어제 어떤 프롬프트로 좋은 결과를 얻었는지, 비슷한 요청을 몇 번이나 반복하고 있는지 알 길이 없습니다. 코드에는 git 히스토리가 있는데, 그 코드를 만들게 한 프롬프트에는 히스토리가 없다는 게 이상했습니다.

정확히는, 히스토리가 없는 게 아니라 흩어져 있습니다. Claude Code는 `~/.claude/projects/`에 세션별 JSONL을 남기고, Codex는 `~/.codex/history.jsonl`에 쌓고, 나머지도 제각각 어딘가에 기록합니다. 데이터는 이미 다 있는데 형식이 전부 다르고, 아무도 모아서 보여주지 않을 뿐입니다. 그래서 모아주는 도구를 만들었습니다. [Oh My Prompt](https://github.com/jiunbae/oh-my-prompt)입니다.

## 첫 프롬프트가 뭐였는지 정확히 아는 이유

이 프로젝트는 2월 2일 밤에 `create-next-app`으로 시작했습니다. 그날 밤 "Prompt Analytics Dashboard"라는 이름으로 첫 커밋을 하고, 새벽 3시 15분까지 홈랩 K8s 배포 파이프라인까지 붙였습니다. 처음에는 공개할 생각 없이 제 프롬프트나 들여다보려던 물건이었습니다.

이름이 바뀐 건 2월 5일입니다. 오후 5시 42분에 Codex에게 이렇게 보냈습니다.

> "저는 현재의 prompt manager repo를 oh-my-prompt라는 이름으로 변경하고 이를 위해 서비스를 향상시킬 계획입니다. 이 서비스의 최종 목적은 (...) 내가 어떤 프롬프트를 어떻게 쓰고 있는지 기록·저장하고 여기서 인사이트를 얻어서 스스로 개선할 수 있도록 하는 것입니다."

31분 뒤에 `Rename to oh-my-prompt and add prompt insights` 커밋이 찍혔습니다. 제가 이 타임스탬프를 분 단위로 아는 이유는 간단합니다. 그 프롬프트가 `~/.codex/history.jsonl`에 남아 있었기 때문입니다. 이 도구가 하려는 일이 정확히 이겁니다. 지금은 글 쓰려고 `jq`로 뒤졌지만, 앞으로는 뒤지지 않아도 되게 만드는 것.

## 캡처: 훅에 얹기

프롬프트를 자동으로 캡처하려면 에이전트와 개발자 사이 어딘가에 끼어들어야 합니다. 프록시 서버로 API 트래픽을 뜨는 방법은 HTTPS 인증서 문제부터가 과한 복잡성이라 접었고, 결국 각 에이전트가 공식으로 제공하는 훅 시스템에 얹기로 했습니다. Claude Code의 훅, Codex의 `notify`, OpenCode의 플러그인, Gemini CLI의 확장 훅. 훅이 한 번 설치되면 사용자는 아무것도 신경 쓸 필요가 없고, 프롬프트는 로컬 SQLite에 조용히 쌓입니다.

"공식 지원"이라고 해서 곱게 됐다는 뜻은 아닙니다. 2월 7일에는 Codex의 notify 설정이 JSON 배열이 아니라 문자열이어야 한다는 걸 알아내느라 커밋을 하나 태웠고, 에이전트마다 이런 자잘한 형식 차이가 계속 나왔습니다. 네 에이전트를 지원한다는 건 이 자잘함을 네 배로 갖고 간다는 뜻입니다.

## MinIO를 넣었다가 일주일 만에 걷어냈습니다

지금 구조는 단순합니다. 클라이언트는 SQLite에 즉시 저장하고, `omp sync`가 500건 단위 청크로 서버 PostgreSQL에 올립니다. 각 레코드에 `content_hash`가 있어서 몇 번을 다시 올려도 중복 저장되지 않습니다.

처음부터 이랬던 건 아닙니다. 초기 설계에는 MinIO가 있었습니다. 클라이언트가 JSONL 청크를 S3 SigV4 서명까지 해가며 오브젝트 스토리지에 올리고, 서버가 그걸 다시 읽는 구조였습니다. 2월 3일에 SigV4 서명과 2-pass 동기화 코드를 리뷰 반영까지 해가며 다듬었는데, 일주일 뒤인 2월 10일에 전부 걷어냈습니다. 커밋 메시지 그대로 옮기면 `refactor: remove MinIO dependency, use PostgreSQL as sole storage backend`. 프롬프트 텍스트를 나르는 데 오브젝트 스토리지 계층이 하나 더 있을 이유가 없었습니다. 홈랩에 MinIO가 이미 떠 있으니까 썼던 것뿐인데, 공개 도구로 만들려니 사용자에게 "PostgreSQL이랑 MinIO를 준비하세요"라고 말해야 한다는 걸 그제야 실감했습니다.

이 결정 이후로 원칙이 하나 생겼습니다. 클라이언트 쪽 의존성은 `npm install`로 끝나야 한다는 것. SQLite는 `better-sqlite3`로 함께 설치되고, 데이터는 파일 하나에 담깁니다. 비행기 안에서도 캡처는 계속 되고, 나중에 `omp sync` 한 번이면 서버에 올라갑니다.

## 공개하던 날의 삽질

2월 9일에 홈랩 Gitea에서 GitHub 공개 저장소로 옮겼습니다. 이날 커밋 로그가 좀 부끄럽습니다. 하드코딩된 개인 도메인과 세션 토큰을 스크럽하는 보안 커밋이 하나, 그리고 npm 배포가 계속 실패해서 15:08, 15:09, 15:11에 연속으로 세 번 찍힌 버저닝 수정 커밋. 버전을 `2026.209.0` 같은 날짜 기반으로 쓰기로 했는데, 태그 패턴과 publish 워크플로우가 이걸 semver로 제대로 넘기지 못해서 3분 간격으로 고치고 푸시하기를 반복했습니다. 사설 인프라에서만 굴리던 프로젝트를 공개로 전환하는 건 코드보다 이런 주변부가 더 오래 걸립니다.

## CLI 첫인상 고치기

2월 말까지는 서버 쪽을 주로 만졌습니다. 2월 18일 저녁에는 품질 점수 레이더 차트, 세션 타임라인 히트맵, 자동 동기화 데몬, 시맨틱 검색 같은 PR 8개를 한 번에 머지했고, 2월 24일부터 28일까지는 R4에서 R7까지 리뷰-수정 라운드를 돌며 IDOR 같은 보안 구멍을 메웠습니다.

그러고 나서 CLI를 다시 보니, 셋업 경험이 형편없었습니다. Node의 `readline`으로 만든 위저드는 동작은 했지만 스피너도 색도 진행감도 없었습니다. 2월 28일에 [`@clack/prompts`](https://github.com/bombshell-dev/clack)로 갈아엎었습니다.

```
$ omp setup

┌  oh-my-prompt
│
◆  Authentication
│  ● Login with email & password
│  ○ Paste existing API token
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
└  Run omp backfill to import existing prompts
```

기술적으로 걸리는 지점이 하나 있었는데, `@clack/prompts`는 ESM-only이고 이 CLI는 `better-sqlite3` 때문에 CJS입니다. 전체를 ESM으로 전환하는 건 리스크 대비 이득이 맞지 않아서, 동적 `import()`로 두 세계를 분리했습니다.

```javascript
// src/omp/ui.js — CJS 파일
const pc = require("picocolors"); // CJS → 동기 로드

let _clack = null;
async function loadClack() {
  if (!_clack) {
    _clack = await import("@clack/prompts"); // ESM → 필요할 때만
  }
  return _clack;
}
```

`picocolors`는 어디서든 동기적으로 쓰고, clack은 인터랙티브 프롬프트가 실제로 필요한 순간에만 로드합니다. 비대화형 모드에서는 ESM 패키지가 아예 로드되지 않습니다.

## 3월 9일 밤

이 글을 쓰기 전날 밤의 커밋 로그입니다.

```
22:01  보안 하드닝 (공개 저장소 대비 마지막 점검)
23:06  zero-dependency 로컬 대시보드 (SQLite)
23:09  로컬 서버 --host 플래그
23:46  OpenCode·Gemini backfill 지원
23:53  backfill 후 sync 커서 자동 리셋
00:22  (자정 넘어) LLM 인사이트에 Gemini 프로바이더 추가
```

이날 밤에 만든 것 중 제일 마음에 드는 건 로컬 대시보드입니다. 그전까지 대시보드를 보려면 `omp serve`로 Docker 컨테이너를 띄워야 했는데, npm으로 CLI를 설치한 사람에게 Docker까지 요구하는 건 진입 장벽이 너무 높았습니다. 그래서 Node 내장 `http` 모듈만으로 돌아가는 모드를 만들었습니다. React도 번들러도 없이 HTML 문자열을 직접 조립하는 방식이지만, CLI가 이미 쓰고 있는 SQLite를 그대로 읽기 때문에 동기화 자체가 필요 없습니다. `omp serve --local` 한 줄이면 브라우저에서 열립니다.

backfill은 도구를 설치하기 전의 기록을 살리는 명령입니다. 훅은 설치 시점 이후만 잡으니까, 이미 몇 달치 쌓여 있는 각 에이전트의 로그를 임포트해야 데이터가 의미 있어집니다. 문제는 넷의 형식이 전부 다르다는 것.

| 에이전트 | 데이터 위치 | 형식 |
|----------|-------------|------|
| Claude Code | `~/.claude/projects/` | 세션별 JSONL |
| Codex | `~/.codex/history.jsonl` + `sessions/` | JSONL |
| OpenCode | `opencode.db` | SQLite |
| Gemini CLI | `~/.gemini/tmp/*/chats/` | 프로젝트 해시별 JSON |

Gemini는 메시지 content가 문자열일 때도 있고 `[{text: "..."}]` 배열일 때도 있어서 둘 다 처리해야 했고, OpenCode는 SQLite DB를 직접 열어 조인해야 했는데 `better-sqlite3`를 이미 쓰고 있던 게 다행이었습니다. 23:53의 커밋은 backfill의 함정 하나를 메운 겁니다. 백필된 레코드는 과거 타임스탬프를 갖기 때문에 동기화 커서보다 뒤에 있어서 `omp sync`가 그냥 지나칩니다. 백필로 새 레코드가 들어오면 커서를 자동으로 리셋하게 했습니다. `content_hash` 중복 감지가 있으니 커서를 리셋해도 같은 프롬프트가 두 번 저장되지는 않습니다.

## 5주, 커밋 128개

오늘 README를 업데이트하면서 세어 보니 2월 2일부터 커밋이 128개입니다. 시작할 수 있는 상태까지는 왔습니다.

```bash
npm install -g oh-my-prompt
omp setup       # 에이전트 자동 감지, 훅 설치
omp backfill    # 과거 기록 임포트
omp serve --local
```

솔직하게 말하면, "기록하고 모아서 보여주는 것"까지가 지금 잘하는 일입니다. 프롬프트 품질 점수도 만들었고 레이더 차트로 예쁘게 그려주기까지 하는데, 정작 저는 그 점수를 아직 별로 믿지 않습니다. "좋은 프롬프트"의 기준을 정의하는 것 자체가 열린 문제라서, 지금 점수는 구조나 구체성 같은 휴리스틱의 조합에 가깝습니다. 데이터가 더 쌓이면 기준부터 다시 잡아야 할 겁니다.

그래도 하나는 분명합니다. 이 5주 동안 제가 에이전트에게 보낸 프롬프트들도 전부 이 DB에 쌓이고 있습니다. 2월 5일의 첫 프롬프트를 `jq`로 뒤져서 찾던 짓은 이제 안 해도 됩니다. 측정할 수 없는 것은 개선할 수 없고, 이건 그 측정의 첫걸음입니다.

---

**프로젝트**: [jiunbae/oh-my-prompt](https://github.com/jiunbae/oh-my-prompt) | [npm](https://www.npmjs.com/package/oh-my-prompt)
