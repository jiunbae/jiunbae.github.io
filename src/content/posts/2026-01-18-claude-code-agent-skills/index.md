---
title: "Claude Code 스킬 시스템 소개와 개인화 경험"
description: "Claude Code의 커스텀 스킬 기능을 활용해 나만의 개발 워크플로우를 만든 경험을 공유합니다"
date: 2026-01-18
permalink: /claude-code-agent-skills
tags: [Claude, AI, Productivity, DevOps]
published: true
---

# Claude Code 스킬 시스템 소개와 개인화 경험

Claude Code를 사용하시는 분들 중에 커스텀 스킬 기능을 아직 활용하지 않고 계신 분들이 많은 것 같습니다. 이 글에서는 Claude Code의 스킬 시스템이 무엇인지 소개하고, 제가 이를 개인화해서 사용하고 있는 방법을 공유해보려 합니다.

## Claude Code의 스킬 시스템이란

Claude Code는 `.claude/skills/` 디렉토리에 마크다운 파일을 두면 이를 스킬로 인식합니다. 스킬은 Claude가 특정 작업을 수행할 때 참고하는 지침서라고 생각하시면 됩니다. (자세한 내용은 [공식 문서](https://code.claude.com/docs/ko/skills)를 참고하세요.)

```
.claude/
└── skills/
    └── my-skill/
        └── SKILL.md
```

스킬 파일에는 다음과 같은 내용을 정의할 수 있습니다:

- 언제 이 스킬이 활성화되어야 하는지 (트리거 키워드)
- 작업을 수행하는 절차
- 주의사항이나 제약조건
- 참고할 스크립트나 템플릿

예를 들어, "커밋해줘"라고 말했을 때 항상 특정 형식의 커밋 메시지를 생성하도록 하거나, 커밋 전에 보안 검사를 수행하도록 지시할 수 있습니다.

## AI 코딩 도구들의 트렌드: 커스텀 지침

흥미로운 점은 이런 커스텀 지침 기능이 Claude Code만의 것이 아니라는 점입니다. 주요 AI 코딩 도구들이 모두 비슷한 기능을 제공하고 있습니다.

| 도구 | 지침 파일 | 특징 |
|-----|----------|------|
| [Claude Code](https://code.claude.com/docs/ko/skills) | `.claude/skills/SKILL.md` | 트리거 키워드, 스크립트 포함 가능 |
| [OpenAI Codex](https://developers.openai.com/codex/guides/agents-md) | `AGENTS.md` | 글로벌/프로젝트 스코프 계층 구조 |
| [Gemini CLI](https://github.com/google-gemini/gemini-cli) | `GEMINI.md` | `/memory` 명령으로 컨텍스트 관리 |
| [GitHub Copilot](https://docs.github.com/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot) | `.github/copilot-instructions.md` | 조직 단위 지침 지원 |

이는 AI 코딩 도구들이 **"일반적인 도우미"에서 "개인화된 협업 도구"로 진화**하고 있다는 신호입니다. 단순히 코드를 생성하는 것을 넘어, 각 개발자의 워크플로우와 선호도에 맞춰 동작하는 방향으로 발전하고 있습니다.

## 왜 스킬을 개인화하게 되었나

Claude Code를 사용하면서 매 세션마다 같은 지시를 반복하고 있다는 것을 깨달았습니다.

```
"커밋 메시지는 한글로 작성해줘"
"API 키 같은 민감 정보는 커밋하면 안 돼"
"이 프로젝트는 이런 구조로 되어 있어..."
```

이런 지시들을 스킬로 정리해두면, 매번 설명할 필요 없이 Claude가 제 작업 방식을 이해하고 따라줍니다. 결국 스킬 개인화는 **Claude를 나에게 맞추는 과정**입니다.

## Agent Skills 프로젝트

이렇게 만든 스킬들을 정리한 것이 [Agent Skills](https://github.com/jiunbae/agent-skills) 프로젝트입니다. 현재 33개 이상의 스킬이 있고, 제 개발 워크플로우에 맞춰 계속 발전시키고 있습니다.

### 스킬 카테고리

| 카테고리 | 스킬 수 | 용도 |
|---------|--------|------|
| development | 7 | Git 커밋, 코드 리뷰, 테스트 |
| context | 3 | 프로젝트 컨텍스트 관리 |
| agents | 6 | 멀티 에이전트 협업 |
| integrations | 8 | Slack, K8s 등 외부 서비스 |
| ml | 4 | ML 모델 배포 |
| business | 3 | 문서 분석 |
| security | 1 | 보안 감사 |
| meta | 2 | 스킬 관리 |

모든 스킬을 사용할 필요는 없습니다. 저도 주로 사용하는 것은 일부이고, 나머지는 필요할 때 꺼내 쓰는 정도입니다.

## 제가 자주 쓰는 스킬들

### git-commit-pr: 커밋 워크플로우

가장 많이 사용하는 스킬입니다. "커밋해줘"라고 말하면 다음 과정을 자동으로 수행합니다:

1. 변경사항 수집
2. 민감 정보 검사 (API 키, 인증 정보 등)
3. 최근 커밋 스타일 참고
4. 커밋 메시지 생성 및 실행

특히 보안 검사 기능이 유용합니다. `.env` 파일이나 API 키 패턴(`sk-*`, `AKIA*`, `ghp_*` 등)을 감지하면 커밋을 중단하고 경고해줍니다. Kubernetes Secret 파일에 템플릿 값(`CHANGE_ME_*`)이 남아있는지도 확인합니다.

실수로 AWS 키를 커밋할 뻔한 적이 있었는데, 이 기능 덕분에 사전에 막을 수 있었습니다. 한 번 세팅해두면 매번 신경 쓰지 않아도 되니 마음이 편합니다.

### context-manager: 에이전트 간 컨텍스트 공유

이 스킬은 단순히 문서를 읽는 것을 넘어, **에이전트들이 서로 소통할 수 있는 공유 공간**을 만들어줍니다.

프로젝트의 `.context/` 디렉토리에 마크다운 형태로 컨텍스트를 저장하면, 메인 에이전트뿐 아니라 서브에이전트들도 이를 참조해서 작업을 진행합니다. 예를 들어 기획 에이전트가 `.context/planning/`에 기획안을 저장하면, 구현 에이전트가 이를 읽고 작업을 수행하는 식입니다.

```
.context/
├── planning/
│   └── auth-implementation.md    # 기획 에이전트가 작성
├── architecture/
│   └── system-design.md          # 아키텍처 결정 기록
├── operations/
│   └── known-issues.md           # 알려진 이슈 공유
└── feedback/
    └── review-comments.md        # 리뷰 피드백 누적
```

작업이 끝나면 컨텍스트를 자동으로 업데이트합니다. "인증 기능 구현 완료"라고 하면 `planning/`의 상태가 업데이트되고, 다음 세션에서 이를 참조할 수 있습니다. 세션 간 연속성이 생기는 셈입니다.

키워드 기반 매칭도 지원합니다. "로그인 버그 수정해줘"라고 요청하면 "로그인", "인증" 관련 문서를 자동으로 찾아서 로드합니다. 매번 프로젝트 구조를 설명할 필요가 없어졌습니다.

### pr-review-loop: 리뷰 자동 반영

PR을 올리고 리뷰를 기다리는 동안 다른 작업을 하고 싶을 때 유용합니다. "리뷰 대기해줘"라고 말하면 Claude가 백그라운드에서 리뷰를 감시하고, 리뷰가 달리면 자동으로 수정 사항을 반영합니다.

```
PR 생성 → 리뷰 대기 → 리뷰 감지 → 분석 → 코드 수정 → 커밋 → 리뷰 재요청 → 반복
```

특히 GitHub Copilot 리뷰와 함께 사용하면 효과적입니다. PR을 올리고 "리뷰 대기해줘"만 하면, Copilot이 리뷰를 달고 Claude가 수정하고 다시 리뷰를 요청하는 루프가 자동으로 돌아갑니다. 최종적으로 "LGTM"이 나오면 루프가 종료됩니다.

물론 아키텍처 변경 같은 큰 결정은 자동으로 처리하지 않고 보고만 합니다. 어디까지나 단순 피드백 반영에 유용한 스킬입니다.

### whoami: 개인 프로필

제 개발 선호도를 정의해둔 스킬입니다. 사용하는 언어, 커밋 메시지 스타일, 코드 주석 언어 등을 기록해두면 Claude가 이에 맞춰 작업합니다.

```markdown
# Developer Profile

## Preferences
- Commit messages: Korean
- Code comments: English
- Primary languages: Python, TypeScript
- Framework preferences: FastAPI, Next.js
```

한 번 설정해두면 모든 프로젝트에서 일관된 스타일을 유지할 수 있습니다.

### planning-agents: 다중 관점 기획

큰 기능을 기획할 때 여러 관점의 의견을 수집하고 싶을 때 사용합니다. "로그인 기능을 3명이 기획해줘"라고 하면 3개의 에이전트가 독립적으로 기획한 후 결과를 병합합니다.

각 에이전트는 서로의 기획을 모르는 상태에서 작업하기 때문에, 다양한 접근 방식이 나옵니다. 최종적으로 공통점과 차이점을 분석한 통합 기획안이 생성됩니다. 혼자 생각할 때보다 놓치는 부분이 줄어듭니다.

## 서브 에이전트 활용 패턴

Agent Skills의 강점 중 하나는 **여러 에이전트를 조합해서 복잡한 작업을 처리**할 수 있다는 점입니다.

### codex-implementer: Claude + Codex 협업

Claude가 오케스트레이터 역할을 하고, [OpenAI Codex CLI](https://developers.openai.com/codex/guides/agents-md)가 실제 구현을 담당하는 패턴입니다.

```
사용자 요청 → Claude 분석/설계 → Codex 구현 → Claude 검토 → 완료
```

Claude는 요구사항을 분석하고 작업을 분해하는 데 강점이 있고, Codex는 full-auto 모드로 빠르게 코드를 작성하는 데 강합니다. 각각의 강점을 살리는 구조입니다.

"CRUD API를 만들어줘"라고 요청하면, Claude가 users/posts/comments로 작업을 분해하고 3개의 Codex 인스턴스가 병렬로 구현합니다. 완료 후 Claude가 결과를 검토하고 통합합니다.

### background-implementer: 백그라운드 병렬 구현

컨텍스트 제한이 걱정될 때 유용한 패턴입니다. 기획 문서를 기반으로 독립적인 작업을 분리하고, 여러 에이전트가 `run_in_background: true` 옵션으로 병렬 실행됩니다.

```
기획 문서 분석 → 작업 분해 → Wave 1 (병렬) → Wave 2 (병렬) → 결과 통합
```

의존성을 고려한 웨이브 단위 실행이 특징입니다:

| Wave | 작업 | 병렬 가능 |
|------|------|----------|
| 1 | DB 마이그레이션, 프론트엔드 타입 | O |
| 2 | 백엔드 모델, 프론트엔드 컴포넌트 | O |
| 3 | API 핸들러 | O |
| 4 | 통합 및 라우팅 | - |

결과는 `.context/impl/` 디렉토리에 저장되어, 나중에 확인하거나 다른 에이전트가 참조할 수 있습니다.

이런 패턴들은 복잡한 기능을 구현할 때 시간을 크게 절약해줍니다. 물론 항상 최종 검토는 필요합니다.

## CLI 도구: 레포별 스킬 관리

Agent Skills는 단순히 스킬 모음이 아니라, **레포지토리별로 필요한 스킬을 설치하고 관리할 수 있는 CLI 도구**를 제공합니다.

### 왜 CLI 도구가 필요한가

모든 프로젝트에 모든 스킬이 필요하지는 않습니다. 웹 프로젝트에서는 `playwright` 스킬이 유용하지만, ML 프로젝트에서는 `triton-deploy` 스킬이 더 필요합니다. CLI 도구를 사용하면 프로젝트 특성에 맞는 스킬만 선택적으로 설치할 수 있습니다.

### 설치 구조

```
~/.claude/skills/           # 전역 스킬 (모든 프로젝트에서 사용)
├── git-commit-pr/
├── context-manager/
└── whoami/

~/my-project/.claude/skills/  # 로컬 스킬 (이 프로젝트에서만 사용)
├── kubernetes-skill/
└── playwright/
```

Claude Code는 로컬 스킬을 먼저 확인하고, 없으면 전역 스킬을 사용합니다. 프로젝트별로 스킬을 오버라이드할 수도 있습니다.

### CLI 사용 예시

```bash
# 전역 설치 (모든 프로젝트에서 사용)
agent-skill install -g git-commit-pr

# 현재 프로젝트에만 설치
cd my-k8s-project
agent-skill init                      # .claude/skills/ 생성
agent-skill install kubernetes-skill  # 로컬 설치

# 설치된 스킬 확인
agent-skill list
agent-skill list --local              # 로컬만
agent-skill list --global             # 전역만
```

### 이 구조의 장점

1. **프로젝트 격리**: 프로젝트마다 필요한 스킬만 활성화
2. **버전 관리**: `.claude/skills/`를 git에 커밋하면 팀원과 스킬 공유 가능
3. **충돌 방지**: 같은 이름의 스킬을 프로젝트별로 다르게 설정 가능
4. **가벼운 시작**: 처음에는 core 스킬만 설치하고, 필요할 때 추가

## 설치 및 사용

관심 있으신 분들을 위해 설치 방법을 간단히 소개합니다.

### 설치

```bash
# 핵심 스킬만 설치
curl -fsSL https://raw.githubusercontent.com/jiunbae/agent-skills/main/setup.sh | bash -s -- --core --cli

# 전체 설치
curl -fsSL https://raw.githubusercontent.com/jiunbae/agent-skills/main/setup.sh | bash -s -- --all --cli
```

### CLI 도구

```bash
agent-skill list              # 설치된 스킬 목록
agent-skill install k8s-skill # 특정 스킬 추가 설치
```

### 프로젝트별 스킬

전역 스킬 외에 프로젝트별로 필요한 스킬만 따로 관리할 수도 있습니다.

```bash
cd my-project
agent-skill init                     # .claude/skills/ 생성
agent-skill install kubernetes-skill # 이 프로젝트에만 설치
```

## 나에게 맞는 스킬 찾기

솔직히 말씀드리면, 제가 만든 스킬들을 그대로 사용하시는 것보다 **본인의 사용 패턴에서 스킬을 발견하는 것**이 더 중요합니다. 개발 습관과 워크플로우는 사람마다 다르기 때문입니다.

첫 스텝으로 추천드리는 방법이 있습니다. `~/.zsh_history`나 `~/.bash_history` 파일을 Claude Code에게 보여주고 "여기서 반복되는 패턴을 찾아서 스킬로 만들 만한 것을 추천해줘"라고 요청해보세요.

```bash
# Claude Code에서 이렇게 시도해보세요
"~/.zsh_history 파일을 분석해서 내가 자주 쓰는 명령어 패턴을 찾아줘.
스킬로 자동화하면 좋을 것 같은 반복 작업이 있으면 추천해줘."
```

저도 이 방법으로 제가 미처 인식하지 못했던 반복 패턴들을 발견했습니다. 예를 들어 `git stash` → 작업 → `git stash pop` 패턴을 자주 쓰고 있다는 걸 알게 되어 이를 스킬로 만들었습니다.

## 스킬 만들기: Claude에게 시키면 됩니다

스킬을 직접 작성할 필요는 없습니다. **Claude에게 바로 만들어달라고 하면 됩니다.**

### 예시 1: 반복 패턴을 스킬로

```
나: "내가 자주 쓰는 패턴이 있어.
    1. git stash로 현재 작업 저장
    2. 다른 브랜치에서 급한 작업
    3. 원래 브랜치로 돌아와서 git stash pop

    이걸 스킬로 만들어서 .claude/skills/에 저장해줘.
    '임시 저장'이라고 말하면 활성화되게 해줘."

Claude: 스킬을 생성했습니다.
        .claude/skills/git-stash-workflow/SKILL.md 파일을 확인해보세요.

        "임시 저장해줘" 또는 "stash 작업"이라고 말하면 활성화됩니다.
```

Claude가 트리거 키워드, 단계별 워크플로우, 주의사항까지 알아서 구성해줍니다.

### 예시 2: history 분석 + 스킬 생성 한 번에

```
나: "~/.zsh_history를 분석해서 내가 자주 쓰는 명령어 패턴 찾아줘.
    반복되는 게 있으면 스킬로 만들 만한지 추천해주고,
    괜찮아 보이는 건 바로 스킬로 만들어줘."

Claude: history를 분석했습니다. 발견한 패턴들:

        1. docker compose down && docker compose up -d (47회)
           → 스킬 추천: 컨테이너 재시작 자동화

        2. git fetch && git rebase origin/main (32회)
           → 스킬 추천: 메인 브랜치 동기화

        3. pytest && git add . && git commit (28회)
           → 스킬 추천: 테스트 후 커밋

        1번과 2번을 스킬로 생성할까요?

나: "응, 둘 다 만들어줘"

Claude: 두 개의 스킬을 생성했습니다.
        - .claude/skills/docker-restart/SKILL.md
        - .claude/skills/sync-main/SKILL.md
```

### 예시 3: 기존 워크플로우 설명하고 스킬화

```
나: "우리 팀은 PR 올리기 전에 항상 이 과정을 거쳐.
    1. npm run lint로 린트 체크
    2. npm run test로 테스트
    3. 둘 다 통과하면 커밋
    4. PR 생성할 때 템플릿 따르기

    이거 스킬로 만들어줘. 린트나 테스트 실패하면 멈추고 알려줘야 해."

Claude: PR 준비 스킬을 생성했습니다.
        .claude/skills/pr-prep/SKILL.md

        "PR 준비해줘" 또는 "PR 올릴게"라고 말하면 활성화됩니다.
        린트/테스트 실패 시 중단하고 실패 내용을 보고합니다.
```

이처럼 본인의 워크플로우를 자연어로 설명하면, Claude가 적절한 구조의 스킬로 만들어줍니다. 스킬 문법을 몰라도 됩니다.

앞서 소개한 history 분석과 조합하면, **패턴 발견부터 스킬 생성까지 한 번에** 처리할 수 있습니다.

## 스킬 개인화를 통해 배운 것들

### Claude와의 협업 방식 정립

스킬을 만들면서 제가 Claude에게 어떤 작업을 맡기고 싶은지, 어떤 방식으로 협업하고 싶은지 명확해졌습니다. 막연하게 "도와줘"라고 하는 것보다 구체적인 절차를 정의해두니 결과물의 일관성이 높아졌습니다.

### 반복 작업의 자동화

매번 같은 지시를 하는 대신 스킬로 정의해두니 시간이 절약됩니다. 특히 보안 검사처럼 빠뜨리기 쉬운 작업을 자동화한 것이 도움이 됐습니다.

### 토큰 효율성

스킬에 스크립트를 포함시켜서 반복되는 도구 호출을 줄였습니다. 예를 들어 `git status`, `git diff`, `git log`를 각각 호출하는 대신 하나의 스크립트로 묶으니 토큰 소모가 줄었습니다.

## 마치며

Claude Code의 스킬 시스템은 AI 도구를 본인의 작업 방식에 맞게 조정할 수 있는 좋은 방법입니다. 처음에는 간단한 커밋 도우미로 시작했지만, 사용하면서 느낀 불편함을 계속 반영하다 보니 지금의 형태가 되었습니다.

제가 만든 스킬들이 모든 분께 맞지는 않겠지만, 스킬 시스템을 어떻게 활용할 수 있는지 참고가 되셨으면 합니다. 본인만의 스킬을 만들어보시는 것도 좋은 경험이 될 것입니다.

---

**이 글에서 소개한 프로젝트**: [jiunbae/agent-skills](https://github.com/jiunbae/agent-skills)
