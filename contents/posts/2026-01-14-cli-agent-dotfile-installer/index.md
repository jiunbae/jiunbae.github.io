---
title: "CLI 에이전트가 dotfile 설치를 쉽고 재미있게 만들었다"
description: "한 줄 명령어로 환경 설정 자동화하기"
date: 2026-01-14
slug: /cli-agent-dotfile-installer
tags: [dev, dotfiles, automation, cli]
published: true
---

GitHub: <https://github.com/jiunbae/settings>

새 컴퓨터 사면 가장 먼저 하는 일이 dotfile 세팅이다. git clone하고 스크립트 돌리고, 이런 저런 설정 파일 심링크 걸고... 몇 번 반복하다 보면 귀찮음이 몰려온다. 그래서 만들었다. 한 줄 명령어로 끝나는 dotfile 설치기를.

## 핵심 기능

- **원라인 설치**: `curl ... | bash`로 끝난다
- **플랫폼 지원**: macOS, Linux, WSL 다 된다
- **모듈식 구조**: 필요한 것만 설치 가능
- **진행 상황 표시**: 어디서 막혔는지 알 수 있다
- **안전함**: 여러 번 실행해도 괜찮다

## 설치해보는 법

```bash
# 전체 설치
curl -fsSL https://raw.githubusercontent.com/jiunbae/settings/master/bootstrap.sh | bash -s -- --all

# 필요한 것만
curl -fsSL https://raw.githubusercontent.com/jiunbae/settings/master/bootstrap.sh | bash -s -- zsh nvim tmux tools
```

터미널에서 위 명령어 치면, 이런 애니메이션 보인다:

```
╔══════════════════════════════════════════════════════════════╗
║  Settings Installer                                          ║
╠══════════════════════════════════════════════════════════════╣
║ [████████████████████████████████████████░░░░░░░░░░]  80%    ║
║  [6/8] Rust toolchain                                        ║
╚══════════════════════════════════════════════════════════════╝

  ✓ Installing Rust via rustup
  ✓ Installing cargo-binstall
  ⠋ Installing eza...
```

## 무엇을 설치하나

| 구성 요소 | 설명 |
|-----------|------|
| **zsh** | Zsh + zinit + Powerlevel10k |
| **nvim** | NeoVim + LazyVim |
| **tmux** | tmux + TPM (Tmux Plugin Manager) |
| **rust** | Rust toolchain + cargo-binstall |
| **uv** | 빠른 Python 패키지 매니저 (pip 대체) |
| **tools** | eza, fd, bat, ripgrep, fzf |
| **tools-extra** | delta, dust, pros, bottom |
| **hishtory** | 크로스 디바이스 히스토리 동기화 |

## 도구들

대체할 도구들:

- `ls` → `eza` (아이콘, git 통합)
- `cat` → `bat` (문법 하이라이팅)
- `find` → `fd` (빠르고 간단)
- `grep` → `ripgrep` (정규식 빠르게)
- `htop` → `bottom` (현대적 UI)

## 구조

3000줄 정도의 bash 스크립트다. 크게 나누면:

- `lib/` - 로깅, 스피너, 유틸리티
- `modules/` - 각 구성 요소별 설치 로직
- `configs/` - 설정 파일들 (.zshrc, .tmux.conf, nvim)
- `scripts/` - 빌드 스크립트

설치 과정은 이렇게 간단하다:

1. 플랫폼 감지
2. 패키지 매니저 세팅
3. 선택된 모듈 순서대로 설치
4. 진행 상황 실시간 표시

## 디테일

**드라이런 모드**: `-n` 또는 `--dry-run`으로 실제 설치 없이 미리보기 가능

```bash
./install.sh --dry-run --all
```

**강제 재설치**: `-f`로 이미 설치된 것도 다시 설치

```bash
./install.sh --force zsh
```

**로그**: `~/.install.log`에 모든 기록 남음

**멱등성**: 같은 스크립트 여러 번 실행해도 괜찮다. 이미 설치된 건 건너뛰고, 설정 파일은 백업 후 덮어씀

## 삽질 기록

처음엔 단순하게 복사만 하는 스크립트였다. 그런데 macOS와 Linux가 패키지 매니저가 달라서 각각 처리해야 했다. brew를 쓰나 apt를 쓰나, 결국 다르다. 그래서 `platform.sh`를 만들어서 추상화했다.

또 진행 상황이 안 보이는 게 답답했다. 설치가 어디서 멈췄는지 알 수 없으니까. 그래서 스피너와 프로그레스 바를 추가했다. `printf`로 커서 조작하면 깔끔하게 나온다.

hishtory랑도 전쟁 좀 했다. Ctrl+R 바인딩이 계속 충돌해서, 결국 native shell integration 방식으로 바꿨다. 이제 `HISHTORY_TERM_INTEGRATION` 환경 변수로 처리한다.

tmux는 catppuccin 테마 입히고 true color 지원 추가했다. status bar에 현재 경로, git 상태, 시간 같은 정보 다 보여주게 만들었다.

## 맺음말

새 컴퓨터 받으면 그냥 curl 명령어 하나 치면 된다. 터미널에서 티키타카 돌아가는 걸 보고 있으면 묘하게 뿌듯하다. 5분 뒤면 zsh, nvim, tmux, git까지 다 준비된다.

이제 삽질할 일이 줄었다. 대신 새로운 환경에서 딱히 설정할 게 없어서 허전할지도. 뭐, 삽질을 줄이고 실제 일을 더 할 수 있으니까 나쁘지 않다.

참고: <https://github.com/jiunbae/settings>
