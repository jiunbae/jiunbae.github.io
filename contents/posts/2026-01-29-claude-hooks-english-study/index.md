---
title: "Claude Code Hooks로 영어 공부하기"
description: "Claude Code의 Hook 시스템을 활용해 코딩하면서 자연스럽게 영어 표현을 익히는 방법을 소개합니다"
date: 2026-01-29
slug: /claude-hooks-english-study
tags: [Claude, AI, Productivity, English]
published: true
---

# Claude Code Hooks로 영어 공부하기

![English Coach Hook 동작 화면](./hook-usage.png)

개발하면서 영어 공부도 하고 싶다는 생각, 해보신 적 있으신가요? Claude Code를 사용하다 보면 프롬프트를 영어로 작성할 일이 많은데, 매번 "이 표현이 맞나?" 고민이 됩니다. 이번 글에서는 Claude Code의 Hook 시스템을 활용해서, 프롬프트를 입력할 때마다 자동으로 영어 교정과 어휘 학습을 해주는 환경을 만든 경험을 공유합니다.

## Claude Code의 Hook 시스템이란

Hook은 Claude Code에서 특정 이벤트가 발생할 때 자동으로 실행되는 명령입니다. `~/.claude/settings.json`에 설정하며, 다양한 이벤트 타입을 지원합니다.

| 이벤트 | 실행 시점 |
|--------|----------|
| `UserPromptSubmit` | 사용자가 프롬프트를 제출할 때 |
| `PreToolUse` | 도구 실행 전 |
| `PostToolUse` | 도구 실행 후 |
| `Notification` | 알림 발생 시 |
| `Stop` | 응답 완료 시 |

Hook 타입도 여러 가지입니다:

- **`command`**: shell 명령을 실행하고 출력을 Claude에 전달
- **`prompt`**: 별도 LLM이 프롬프트를 평가하고 결과를 전달
- **`agent`**: 에이전트가 검증 작업을 수행

영어 코칭에는 `command` 타입이 적합합니다. shell 스크립트가 지시를 출력하면 Claude가 이를 수신하고 영어 교정을 수행합니다. 설정이 간단하고 안정적으로 동작합니다.

## 영어 코칭 Hook 만들기

### 아이디어

매 프롬프트 제출 시 자동으로:

1. 내가 입력한 영어(또는 한글) 프롬프트를 **자연스러운 영어 표현**으로 다시 써주고
2. 어려운 단어의 **한국어 뜻**을 함께 보여주는 것

코딩하면서 영어 표현을 자연스럽게 습득하는 환경을 만드는 겁니다.

### 1단계: Hook 스크립트 작성

먼저 `~/.claude/hooks/english-coach.sh` 파일을 만듭니다:

```bash
#!/bin/bash
cat <<'INSTRUCTION'
[English Coach] Before responding to the user's task, do the following FIRST:

1. **Improved English**: Rewrite the user's prompt in natural, commonly-used
   English. Preserve the original intent completely. If the prompt is in Korean,
   translate it. If already in English, improve it.
2. **Vocabulary**: List any intermediate/advanced English words from the improved
   version with Korean definitions.

Format:
---
> **Your prompt in better English:**
> (rewritten prompt here)
>
> **Vocabulary:**
> - word1: Korean meaning
> - word2: Korean meaning
---

Then proceed to actually handle the user's request as normal.
INSTRUCTION
```

실행 권한을 부여합니다:

```bash
chmod +x ~/.claude/hooks/english-coach.sh
```

### 2단계: settings.json에 등록

`~/.claude/settings.json`에 hook을 추가합니다:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash ~/.claude/hooks/english-coach.sh",
            "statusMessage": "English coaching..."
          }
        ]
      }
    ]
  }
}
```

`UserPromptSubmit` 이벤트에 `command` 타입으로 등록하면, 프롬프트를 제출할 때마다 스크립트가 실행되고 출력이 Claude에게 전달됩니다.

### 실제 동작 예시

![English Coach Hook 동작 화면](./hook-usage.png)

프롬프트를 입력하면:

```
> i want to magnage this hooks in current repo and install with install script
```

Claude가 응답하기 전에 먼저 이런 코칭이 표시됩니다:

```
---
> **Your prompt in better English:**
> I'd like to manage these hooks in the current repo and set up
> an install script to install them.
>
> **Vocabulary:**
> - manage: 관리하다
> - repository (repo): 저장소
> - install script: 설치 스크립트
---
```

그 다음 원래 요청에 대한 응답이 이어집니다. 자연스럽게 "magnage"가 "manage"로 교정되고, "i want to"가 "I'd like to"로 개선되는 것을 볼 수 있습니다.

한글로 입력해도 동작합니다:

```
> 이 hook은 전역으로 설치되나요?
```

```
---
> **Your prompt in better English:**
> Is this hook installed globally?
>
> **Vocabulary:**
> - globally: 전역적으로, 시스템 전체에 걸쳐
---
```

## Hook 타입 선택: `command` vs `prompt`

Hook 타입을 선택할 때 두 가지를 비교해볼 수 있습니다.

| | `command` 타입 | `prompt` 타입 |
|---|---|---|
| 동작 방식 | shell 스크립트 실행 → 출력을 Claude에 전달 | 별도 LLM이 직접 평가 → 결과를 Claude에 주입 |
| 안정성 | 안정적 | 이벤트에 따라 호환성 이슈 가능 |
| 파일 필요 | 스크립트 파일 필요 | 설정만으로 동작 |
| 비용 | 없음 | 소량의 추가 토큰 사용 |
| 커스터마이징 | 자유도 높음 (shell 활용) | 프롬프트 수정만 가능 |

처음에는 `prompt` 타입이 LLM이 직접 평가하니 더 안정적일 거라 생각했지만, `UserPromptSubmit` 이벤트에서는 호환성 이슈가 있었습니다. `command` 타입은 단순히 텍스트를 출력하는 구조라 안정적이고, Claude가 hook 출력을 잘 따라줍니다.

영어 코칭처럼 **매 프롬프트마다 실행되는 hook**에는 `command` 타입이 현재로서는 더 적합합니다.

## Hook을 레포에서 관리하기

설정이 `settings.json`에만 있으면 관리가 어렵습니다. 저는 [Agent Skills](https://github.com/jiunbae/agent-skills) 레포에서 hook을 코드로 관리하고 있습니다.

### 레포 구조

```
hooks/
├── english-coach.sh   # hook 스크립트
└── hooks.json         # hook 레지스트리
```

`hooks.json`이 핵심입니다. 각 hook의 이벤트, 타입, 스크립트를 정의합니다:

```json
{
  "english-coach": {
    "description": "Rewrites prompts in better English with vocabulary",
    "script": "english-coach.sh",
    "event": "UserPromptSubmit",
    "type": "command",
    "statusMessage": "English coaching..."
  }
}
```

### 설치 스크립트

`install.sh --hooks` 명령으로 설치합니다:

```bash
# hooks만 설치
./install.sh --hooks

# Core 스킬 + CLI + Hooks 풀 설치
./install.sh --core --cli --hooks

# hooks 제거
./install.sh --uninstall-hooks
```

설치 스크립트가 하는 일은:

1. `hooks.json` 레지스트리를 읽어서 hook 목록 파싱
2. `command` 타입이면 스크립트 파일을 `~/.claude/hooks/`에 심링크
3. `prompt` 타입이면 스크립트 파일 불필요 (설정만 등록)
4. `~/.claude/settings.json`에 hook 설정을 자동 병합
5. 중복 체크 — 같은 hook이 이미 있으면 건너뜀

제거 시에는 역순으로 스크립트 파일과 settings.json 설정을 깨끗하게 정리합니다.

### 새 hook 추가하기

영어 코칭 외에 다른 hook도 쉽게 추가할 수 있습니다. `hooks.json`에 항목을 추가하고 `./install.sh --hooks`를 다시 실행하면 됩니다.

예를 들어, 커밋 전에 보안 검사를 하는 hook:

```json
{
  "english-coach": { ... },
  "security-check": {
    "description": "Checks for secrets before tool execution",
    "event": "PreToolUse",
    "type": "prompt",
    "prompt": "Check if $ARGUMENTS contains any secrets, API keys, or credentials...",
    "matcher": "Write",
    "statusMessage": "Security check..."
  }
}
```

`matcher` 필드로 특정 도구에서만 실행되도록 필터링할 수도 있습니다.

## 실제 학습 효과

2주 정도 사용해본 소감입니다:

- **자연스러운 노출**: 매 프롬프트마다 교정을 보니, "이 표현은 이렇게 쓰는구나"를 반복적으로 익히게 됩니다
- **실용적 어휘**: 기술 문맥에서의 영어 표현을 배우게 됩니다. "I'd like to" vs "I want to", "set up" vs "setup" 같은 뉘앙스 차이
- **한글 프롬프트도 OK**: 한글로 입력해도 영어 번역을 보여주니, 영어로 어떻게 표현할지 자연스럽게 학습
- **부담 없음**: 별도 시간을 내지 않아도, 코딩하는 동안 자연스럽게 영어 표현에 노출됨

물론 이것만으로 영어 실력이 비약적으로 늘지는 않겠지만, 개발 영어 표현에 대한 감각을 키우는 데는 분명 도움이 됩니다. 특히 영어 문서를 읽거나 이슈를 작성할 때 자연스러운 표현이 떠오르는 경험이 늘었습니다.

## 마무리

Claude Code의 Hook 시스템은 단순한 자동화를 넘어, 개발 워크플로우에 다양한 기능을 플러그인처럼 추가할 수 있는 확장 포인트입니다. 영어 코칭은 그 중 하나의 활용 사례일 뿐이고, 보안 검사, 코드 스타일 강제, 로깅 등 다양한 용도로 활용할 수 있습니다.

코딩하면서 영어 공부도 하고 싶다면, hook을 한번 설정해보세요. 설정 한 번이면 매 프롬프트마다 자연스럽게 영어 표현을 접할 수 있습니다.

---

Hook을 포함한 전체 스킬 시스템에 대해서는 [Claude Code 스킬 시스템 소개와 개인화 경험](/claude-code-agent-skills) 글도 함께 참고해주세요.
