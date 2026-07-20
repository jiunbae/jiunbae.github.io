---
title: "agt: AI 코딩 에이전트를 위한 패키지 매니저"
description: "마크다운 스킬 모음이던 agent-skills가 셸 스크립트 3개를 거쳐 Rust CLI가 되기까지. 석 달 반, 커밋 167개 동안 겪은 파편화·훅 삽질·비밀번호 커밋 사건의 기록입니다."
date: 2026-03-12
permalink: /agt-package-manager-for-ai-agents
tags: [AI, CLI, Claude, Codex, Gemini, OpenSource, Rust, DevTools]
published: true
---

# agt: AI 코딩 에이전트를 위한 패키지 매니저

![agt 배너 — AI 코딩 에이전트를 위한 스킬, 페르소나, 훅을 관리하는 모듈형 툴킷](/images/posts/agt-package-manager-for-ai-agents/banner.png)

처음부터 패키지 매니저를 만들 생각은 없었습니다. 2025년 11월 23일의 첫 커밋은 `agent-skills`라는 이름의, 마크다운 파일 몇 개와 설치 가이드가 전부인 레포였습니다. Claude Code를 쓰다 보니 "이렇게 해줘"라는 지침이 자꾸 쌓였고, 프로젝트마다 복사해 붙이는 게 지겨워서 한군데 모아둔 것뿐이었죠.

문제는 에이전트가 하나가 아니라는 겁니다. Claude Code는 `.claude/skills/`, Codex는 `AGENTS.md`, Gemini CLI는 `GEMINI.md`. 같은 내용을 세 가지 방식으로 관리해야 하고, 저는 데스크탑과 랩탑을 오가며 일합니다. 파일이 다섯 개일 때는 손으로 복사해도 됐는데, 12월 한 달 동안 커밋이 40개 넘게 쌓이면서 상황이 달라졌습니다. 스킬을 하나 고치면 어느 머신의 어느 에이전트에 반영됐는지 저도 모르게 된 거죠. 설정 동기화 문제는 결국 패키지 매니저 문제입니다. npm이 풀던 것과 같은 문제요.

## install.sh의 시대

처음 몇 달은 셸 스크립트로 버텼습니다. `install.sh`가 스킬을 `~/.claude/skills/`로 복사해 주고, 12월 15일에는 Codex CLI 지원도 붙였습니다. 12월 11일에는 `claude-skill`이라는 CLI를 얹었는데, 프롬프트를 주면 Claude가 맞는 스킬을 골라 실행해 주는 물건이었습니다. 만든 그날 저녁에만 옵션 커밋이 다섯 개 쌓였습니다. 시스템 프롬프트 크기 제한에 걸려서 고치고, 스트리밍 출력이 안 나와서 고치고, `--result-only` 옵션을 붙이고. 셸 스크립트에 기능을 하나씩 덧대는 전형적인 경로였죠.

1월 29일에는 훅을 넣었습니다. 첫 훅은 english-coach — 제 프롬프트를 자연스러운 영어로 다듬어 주는 `UserPromptSubmit` 훅이었습니다. 이게 얼마나 삽질이었는지는 커밋 로그가 말해줍니다. 15:56에 추가하고, 17:16 프롬프트 길이 제한 가드, 17:19 콘텐츠 타입 감지, 18:18 멀쩡한 프롬프트를 차단하던 버그 수정, 18:36 JSON stdin 파싱 수정. 훅 하나 넣고 저녁 내내 고쳤습니다. 훅은 스킬과 달리 매 이벤트마다 무조건 실행되니까, 한 번 잘못 만들면 모든 프롬프트가 막힙니다. 이날 배웠습니다.

바로 다음 날인 1월 30일에는 부끄러운 커밋이 하나 있습니다. `security: remove exposed password from playwright scripts`. 스킬에 딸린 스크립트에 비밀번호가 하드코딩된 채 커밋돼 있던 걸 발견해서, 히스토리를 정리하고 그 김에 security-auditor 스킬이 비밀번호와 사용자 경로를 감지하도록 강화했습니다. 보안 감사 스킬을 만들어 놓고 정작 그 스킬이 사는 레포에서 비밀번호가 새고 있었던 겁니다.

2월 초에는 늘리는 대신 줄였습니다. 2월 1일에 당시 37개였던 스킬 전부를 Anthropic 공식 스킬 가이드라인에 맞춰 다시 썼고, 2월 7일에는 그중 6개를 그냥 지웠습니다. 겹치거나, 만들어 놓고 한 번도 안 쓴 것들이었습니다. 스킬은 쌓이는 게 아니라 관리되는 순간부터 자산이 됩니다.

## Rust CLI가 된 날

2월 18일 밤, 페르소나 리뷰용 CLI(`agent-persona`)를 추가했습니다. 이 시점에서 셸 스크립트가 세 벌이 됐습니다. `agent-skill`(설치), `agent-persona`(리뷰), `claude-skill`(실행). 각자 옵션 파싱이 다르고, 에러 처리가 다르고, Windows에서는 셋 다 애매했습니다.

2월 19일 오후에 이걸 전부 하나로 합쳤습니다. 13:27에 Rust CLI 바이너리 커밋, 13:32에 npm 패키징, 13:49에 날짜 기반 버저닝 도입, 14:01에 `jiunbae/agent-skills` → `open330/agt` 리브랜드. CLI 첫 커밋부터 리브랜드까지 34분입니다. 물론 코드는 AI 에이전트가 대부분 썼고, 저는 셸 스크립트 세 벌의 동작을 그대로 옮기는지 지켜보는 쪽이었습니다.

Rust를 고른 이유는 단순합니다. 스킬 관리 도구는 모든 프로젝트, 모든 머신에서 돌아야 하는데, Node나 Python 런타임을 전제할 수 없습니다. 단일 바이너리면 그 문제가 사라집니다. npm으로도 설치되는데(`npm install -g @open330/agt`), 이건 플랫폼별 pre-built 바이너리를 optional dependency로 내려받는 방식이라 결국 실행되는 건 같은 바이너리입니다. 버전을 semver 대신 날짜(`2026.2.19`)로 바꾼 것도 같은 맥락입니다. 스킬 모음에 "하위 호환을 깨는 변경"같은 개념을 유지하는 게 부자연스러워서, 그냥 언제 빌드인지만 남기기로 했습니다.

```console
$ agt --help
agt — A modular toolkit for extending AI coding agents

Commands:
  skill        Manage agent skills
  hook         Manage Claude Code hooks (command, http, prompt, agent)
  team         Manage agent teams (spawn coordinated multi-agent workflows)
  persona      Manage agent personas
  run          Run prompt with skill matching
  completions  Generate shell completion scripts
```

이름이 바뀌면서 커맨드도 정리됐습니다.

| Before | After |
|--------|-------|
| `jiunbae/agent-skills` | `open330/agt` |
| `~/.agent-skills/` | `~/.agt/` |
| `agent-skill install <skill>` | `agt skill install <skill>` |
| `agent-persona review <p>` | `agt persona review <p>` |
| `claude-skill "prompt"` | `agt run "prompt"` |

`~/.claude/skills/` 설치 대상과 SKILL.md 포맷은 그대로 둬서, 기존 사용자는 이름만 바꾸면 됩니다.

낭만은 여기까지고, 그날 밤은 CI와 싸웠습니다. 23:24 릴리스 생성 race condition 수정, 23:44 태그에서 버전 sync, 23:47 npm publish 멱등성 수정. 바이너리를 네 플랫폼(macOS arm64/x64, Linux x64/arm64)으로 빌드해서 GitHub 릴리스와 npm에 동시에 올리는 파이프라인은, 만들어 본 분은 아시겠지만 한 번에 되는 법이 없습니다.

## 이틀 밤의 도그푸딩

2월 20일 밤부터 21일 새벽 2시 32분까지, 버전이 2026.2.22에서 2026.2.28까지 올라갔습니다. 전부 직접 써 보다가 걸린 것들입니다. 그중 하나가 `fix(agt): reject persona names with spaces` 커밋인데, `agt persona review security-reviewer`를 치다가 인자 순서를 헷갈리면 뒤의 단어들이 통째로 페르소나 이름으로 들어가는 문제였습니다. 에러도 안 나고 이상한 파일을 찾다 실패하죠. 이름에 공백이 오면 "인자 순서 확인하세요"라고 거부하게 바꿨습니다.

그리고 agt 자신에게 리뷰를 시켰습니다. `agt persona review`로 자기 코드베이스를 돌린 결과가 `fix(agt): apply review findings — validate_name coverage, tarball limit, clippy` 커밋으로 남아 있습니다. 이름 검증이 안 닿는 경로, 원격 설치 시 tarball 크기 제한 없음 같은 것들이 나왔습니다. 도구가 자기 자신의 구멍을 찾는 건 볼 때마다 묘한 기분입니다.

## 전부 마크다운

CLI는 Rust지만, agt가 관리하는 것들은 전부 순수 마크다운입니다.

```yaml
# security/security-auditor/SKILL.md
---
name: auditing-security
description: Audits repository security by analyzing current code and
  commit history for sensitive information leaks.
trigger_keywords:
  - 보안 검사
  - security audit
---
```

바이너리도 런타임 의존성도 없으니 Claude Code든 Codex든 Gemini든 그냥 읽으면 되고, git으로 히스토리가 남고, 스킬이 이상하게 굴면 파일을 열어서 고치면 됩니다. 에이전트 생태계가 어디로 튈지 모르는 상황에서, 특정 에이전트의 플러그인 포맷에 묶이지 않는 게 이 프로젝트의 몇 안 되는 확신이었습니다.

그 위의 관리 레이어가 npm과 닮은 건 의도한 결과입니다. 프로필은 `dependencies` 목록에 해당합니다. `agt skill install --profile core`를 치면 필수 스킬 7개(커밋/PR 가이드, 컨텍스트 로더, 보안 감사, 백그라운드 구현·기획·리뷰 에이전트 등)가 한 번에 깔립니다. 그룹은 스코프처럼 동작해서 `integrations/slack`처럼 카테고리 단위로 설치·삭제할 수 있고요. 2월 24~25일에는 인터랙티브 TUI 설치기와 `--from`으로 임의의 GitHub 레포에서 설치하는 기능, GitHub 토큰으로 private 레포를 읽는 기능이 붙었습니다. 스킬 레포와 CLI가 분리될 수 있어야 다른 사람이 자기 스킬 레포를 만들 수 있으니까요.

## 페르소나, 팀, 그리고 꺼버린 CI

페르소나는 스킬과 결이 다릅니다. 스킬이 "무엇을 할 것인가"라면 페르소나는 "어떤 눈으로 볼 것인가"입니다. 지금은 보안·아키텍처·코드 품질·성능·DB·프론트엔드·DevOps, 일곱 명의 리뷰어가 있고, `agt persona review security-reviewer --codex`처럼 페르소나마다 다른 LLM을 붙여 병렬로 돌릴 수 있습니다. 결과는 `.context/reviews/`에 라운드 번호로 쌓여서 이전 리뷰와 비교합니다. 첫 페르소나 중 하나가 "도도한 키위새"라는 이름의 Rust 시스템 리뷰어였는데, agt 자신을 리뷰시키려고 만든 거였습니다.

3월 5일에는 훅 관리와 팀 기능을 한 커밋에 넣었습니다. 훅은 지금 11개가 세션 라이프사이클을 덮고 있고(프롬프트 제출 시 영어 코칭, 도구 실행 전 보안 게이트와 커밋 가드, 편집 후 린트 등), 팀은 code-review·feature-dev·debug·refactor·research 다섯 개의 YAML 정의로, 리뷰어 셋이 병렬로 보고 종합하는 식의 멀티 에이전트 워크플로우를 `agt team spawn` 한 줄로 띄웁니다.

솔직한 커밋도 하나 적어두겠습니다. 3월 8일, `chore: remove CI workflows (development phase)`. 하루에 버전이 예닐곱 번 오르는 시기에 릴리스 파이프라인이 커밋마다 도는 건 도움보다 방해였고, 그냥 껐습니다. 릴리스 자동화는 개발이 안정되면 다시 붙일 생각입니다. 2월 19일 밤에 그 고생을 하고 얻은 결론이 "지금은 필요 없다"라는 게 좀 허무하긴 합니다.

## 석 달 반의 숫자

![agt 로고](/images/posts/agt-package-manager-for-ai-agents/logo.png)

첫 커밋부터 오늘까지 커밋 167개. 스킬 36개(9개 카테고리), 리뷰어 페르소나 7명, 훅 11개, 팀 5개, 그리고 이걸 관리하는 Rust 코드 약 6,600줄. 마크다운 모음으로 시작한 레포치고는 멀리 왔습니다.

npm이 Node.js 패키지를 표준화했듯 agt가 에이전트 확장을 표준화할 수 있을지는 — 솔직히 모르겠습니다. 에이전트마다 확장 포맷이 다른 파편화는 여전하고, agt는 그 위를 덮는 제 나름의 레이어일 뿐입니다. 어제도 스코프가 다른 그룹에 같은 이름의 스킬이 있으면 설치가 꼬이는 버그를 고쳤습니다. 다만 확실한 건, 이제 새 머신에서 한 줄이면 제 작업 환경 전체가 복원된다는 겁니다.

```bash
curl -fsSL https://raw.githubusercontent.com/open330/agt/main/setup.sh | bash
```

소스는 [GitHub](https://github.com/open330/agt)에, 바이너리는 [npm](https://www.npmjs.com/package/@open330/agt)에 있습니다.
