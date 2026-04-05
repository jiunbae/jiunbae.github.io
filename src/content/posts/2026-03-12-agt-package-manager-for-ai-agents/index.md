---
title: "agt: AI 코딩 에이전트를 위한 패키지 매니저"
description: "Claude Code, Codex, Gemini CLI 등 AI 코딩 에이전트의 스킬, 페르소나, 훅을 통합 관리하는 CLI 도구 agt를 만든 이야기입니다."
date: 2026-03-12
permalink: /agt-package-manager-for-ai-agents
tags: [AI, CLI, Claude, Codex, Gemini, OpenSource, Rust, DevTools]
published: true
---

# agt: AI 코딩 에이전트를 위한 패키지 매니저

## 들어가며

AI 코딩 에이전트를 쓰다 보면 자연스럽게 커스텀 지침이 쌓입니다. Claude Code에는 `.claude/skills/`, Codex에는 `AGENTS.md`, Gemini CLI에는 `GEMINI.md`. 각 에이전트마다 "이렇게 해줘"라는 지침을 따로 관리해야 합니다.

스킬이 5개일 때는 괜찮았습니다. 10개가 되니 좀 번거로워졌고, 33개가 되니 수동 관리가 불가능해졌습니다. 여기에 리뷰어 페르소나 7개, 이벤트 훅 11개, 팀 워크플로우 5개까지 더하면 — 이건 패키지 매니저가 필요한 규모입니다.

그래서 [agt](https://github.com/open330/agt)를 만들었습니다.

## npm이 패키지를 관리하듯, agt는 에이전트를 관리한다

agt의 핵심 아이디어는 간단합니다: **AI 에이전트의 확장 기능도 패키지처럼 관리할 수 있어야 한다.**

```bash
# npm install처럼, 스킬을 설치한다
agt skill install --profile core

# 리모트 레포에서도 설치 가능
agt skill install -g --from open330/agt

# 설치된 스킬 확인
agt skill list --installed
```

npm이 `node_modules/`에 패키지를 설치하듯, agt는 `~/.claude/skills/`에 스킬을 설치합니다. 프로필(profile)은 `package.json`의 `dependencies` 같은 역할이고, 그룹(group)은 스코프(`@org/package`)에 해당하죠.

## 왜 마크다운인가

agt의 모든 스킬과 페르소나는 **순수 마크다운**입니다. 바이너리도 없고, 런타임 의존성도 없습니다.

```
security/security-auditor/
└── SKILL.md
```

```yaml
# SKILL.md 프론트매터
---
name: security-auditor
description: "Repository security audit. Use for '보안 점검', 'security audit' requests."
trigger-keywords: 보안 점검, security audit, 민감 정보 검사
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
priority: high
tags: [security, audit]
---
```

이 설계에는 이유가 있습니다:

1. **에이전트 독립적**: 마크다운이니 Claude Code든 Codex든 Gemini든 읽을 수 있습니다
2. **버전 관리 가능**: Git으로 히스토리가 추적됩니다
3. **사람이 읽을 수 있음**: SKILL.md를 열면 스킬이 뭘 하는지 바로 이해됩니다
4. **디버깅 가능**: 스킬이 이상하게 동작하면 마크다운을 직접 수정하면 됩니다

## 33개 스킬의 생태계

현재 agt에는 8개 카테고리에 33개 스킬이 있습니다:

| 카테고리 | 스킬 수 | 예시 |
|----------|---------|------|
| agents | 3 | background-implementer, background-planner, background-reviewer |
| development | 7 | git-commit-pr, playwright, pr-review-loop, task-master |
| business | 3 | bm-analyzer, document-processor, proposal-analyzer |
| integrations | 10 | kubernetes-skill, slack-skill, discord-skill, notion-summary |
| ml | 4 | audio-processor, ml-benchmark, model-sync, triton-deploy |
| security | 1 | security-auditor |
| context | 2 | context-manager, static-index |
| meta | 3 | karpathy-guide, skill-manager, skill-recommender |

모든 스킬을 설치할 필요는 없습니다. 프로필로 필요한 것만 골라서 설치합니다:

```bash
agt skill install --profile core    # 필수 7개만
agt skill install --profile dev     # 개발 도구 전체
agt skill install --profile full    # 코어 + 개발 + 에이전트
```

`core` 프로필에는 다음 7개가 포함됩니다:

- `git-commit-pr` — 보안 검증 포함 커밋/PR 가이드
- `context-manager` — 프로젝트 컨텍스트 자동 로더
- `static-index` — 글로벌 정적 컨텍스트 인덱스
- `security-auditor` — 레포지토리 보안 감사
- `background-implementer` — 멀티 LLM 병렬 구현
- `background-planner` — 멀티 LLM 병렬 기획
- `background-reviewer` — 멀티 LLM 병렬 코드 리뷰

## 페르소나: AI에게 전문가의 눈을 빌려주다

스킬이 "무엇을 할 것인가"라면, 페르소나는 "어떤 관점에서 볼 것인가"입니다.

```bash
# 보안 전문가 관점으로 코드 리뷰
agt persona review security-reviewer

# Codex로 아키텍처 리뷰
agt persona review architecture-reviewer --codex

# 스테이징된 변경사항만 리뷰
agt persona review code-quality-reviewer --staged
```

현재 7명의 전문 리뷰어 페르소나가 있습니다:

| 페르소나 | 역할 | 관점 |
|----------|------|------|
| security-reviewer | Senior AppSec Engineer | OWASP, 인증, 인젝션 |
| architecture-reviewer | Principal Architect | SOLID, API 설계, 결합도 |
| code-quality-reviewer | Staff Engineer | 가독성, 복잡도, DRY |
| performance-reviewer | Performance Engineer | 메모리, CPU, I/O |
| database-reviewer | Senior DBA | 쿼리 최적화, 스키마 |
| frontend-reviewer | Senior Frontend Engineer | React, 접근성 |
| devops-reviewer | Senior DevOps/SRE | K8s, IaC, CI/CD |

가장 강력한 기능은 **병렬 멀티 페르소나 리뷰**입니다:

```bash
# 3명의 전문가가 동시에, 각각 다른 LLM으로 리뷰
agt persona review security-reviewer --gemini \
  -o ".context/reviews/R01-security.md" &
agt persona review architecture-reviewer --codex \
  -o ".context/reviews/R01-architecture.md" &
agt persona review code-quality-reviewer --claude \
  -o ".context/reviews/R01-quality.md" &
wait
```

3명의 전문가가 3개의 다른 AI 모델로 동시에 리뷰합니다. 결과는 `.context/reviews/`에 라운드 넘버링으로 저장되어, 이전 리뷰와 비교할 수 있습니다.

## 훅: 이벤트 기반 자동화

agt는 Claude Code의 훅 시스템과 깊이 통합되어 있습니다.

```json
{
  "hooks": [
    {
      "name": "english-coach",
      "event": "UserPromptSubmit",
      "type": "command",
      "description": "프롬프트를 자연스러운 영어로 재작성"
    },
    {
      "name": "security-gate",
      "event": "PreToolUse",
      "type": "prompt",
      "description": "셸 명령 실행 전 보안 검사"
    },
    {
      "name": "commit-guard",
      "event": "PreToolUse",
      "type": "command",
      "description": "민감 파일(.env, 키) 커밋 방지"
    }
  ]
}
```

11개의 훅이 세션의 전체 라이프사이클을 커버합니다:

- **프롬프트 제출 시**: 영어 코칭, 프롬프트 로깅
- **도구 실행 전**: 보안 게이트, 커밋 가드
- **도구 실행 후**: 린터 실행, 도구 분석
- **세션 시작 시**: 프로젝트 컨텍스트 자동 로드
- **팀 활동 시**: 활동 로그, 완료 검증

## 팀: 사전 정의된 협업 워크플로우

혼자 개발하더라도, AI 에이전트들이 팀처럼 협업할 수 있습니다.

```yaml
# teams/code-review.yml
name: code-review
description: "Multi-perspective code review"
tasks:
  - name: security-reviewer
    persona: security-reviewer
    description: "Review for security vulnerabilities"
  - name: architecture-reviewer
    persona: architecture-reviewer
    description: "Review structure and design patterns"
  - name: quality-reviewer
    description: "Review code quality and test coverage"
  - name: synthesize
    depends_on: [security-reviewer, architecture-reviewer, quality-reviewer]
    description: "Merge findings into unified report"
```

5개의 사전 정의된 팀이 있습니다:

- **code-review**: 3명의 리뷰어가 동시 리뷰 → 통합 보고서
- **feature-dev**: 아키텍트 → 구현자 + 테스터 + 문서화 병렬 실행
- **debug**: 로그 분석, 코드 추적, 재현 테스트를 병렬로
- **refactor**: 기획 → 마이그레이션 → 검증 순차 실행
- **research**: 기술/시장/반론 조사를 병렬로 → 종합

## Rust로 만든 CLI

agt CLI는 Rust로 작성되었습니다. 이유는 심플합니다:

1. **빠름**: 스킬 목록 출력이 수 밀리초
2. **단일 바이너리**: Node.js나 Python 런타임 불필요
3. **크로스 플랫폼**: macOS (arm64, x64), Linux (x64, arm64) 지원

```toml
# Cargo.toml
[profile.release]
opt-level = "z"       # 크기 최적화
lto = true            # 링크 타임 최적화
strip = true          # 심볼 제거
codegen-units = 1     # 최적화 극대화
```

npm을 통해서도 설치할 수 있습니다. 플랫폼별 optional dependency로 pre-built 바이너리를 제공해요:

```bash
npm install -g @open330/agt
# 또는 설치 없이
npx @open330/agt skill list
```

## agent-skills에서 agt로

원래 이 프로젝트는 `jiunbae/agent-skills`라는 이름이었습니다. 개인 레포에서 시작해서 스킬 파일만 모아두는 컬렉션이었죠.

그런데 쓰다 보니 설치, 업데이트, 프로필 관리 같은 기능이 필요해졌습니다. 결국 CLI 도구가 되었고, Open330 오거니제이션으로 이전하면서 `agt`라는 이름을 갖게 됐습니다.

| Before | After |
|--------|-------|
| `jiunbae/agent-skills` | `open330/agt` |
| `~/.agent-skills/` | `~/.agt/` |
| `agent-skill install <skill>` | `agt skill install <skill>` |
| `agent-persona review <p>` | `agt persona review <p>` |
| 설치 스크립트만 존재 | Rust CLI + npm 배포 |

바뀌지 않은 것도 있습니다: `~/.agents/` 정적 컨텍스트 경로, `~/.claude/skills/` 설치 대상, SKILL.md 포맷. 기존 사용자는 마이그레이션 가이드(`MIGRATION.md`)를 따라 이름만 바꾸면 됩니다.

## 마치며

AI 코딩 에이전트 생태계가 빠르게 성장하고 있습니다. Claude Code, Codex, Gemini CLI, OpenCode — 각각 장단점이 다르고, 각각의 확장 방식도 다릅니다.

agt는 이 파편화된 생태계 위에 하나의 관리 레이어를 제공하려 합니다. npm이 Node.js 패키지를 표준화했듯이, agt가 AI 에이전트의 확장 기능을 표준화할 수 있을지는 아직 모르겠습니다. 하지만 적어도 제 워크스테이션에서는 잘 동작하고 있습니다. 33개의 스킬과 7명의 리뷰어와 11개의 훅이 매일 돌아가고 있으니까요.

```bash
# 시작은 한 줄이면 됩니다
curl -fsSL https://raw.githubusercontent.com/open330/agt/main/setup.sh | bash
```
