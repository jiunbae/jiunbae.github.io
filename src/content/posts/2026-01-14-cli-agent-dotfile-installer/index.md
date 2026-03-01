---
title: "개인 환경 설정 (dotfiles) 관리하기"
description: "한 줄 명령어로 환경 설정 자동화하기"
date: 2026-01-14
permalink: /cli-agent-dotfile-installer
tags: [dev, dotfiles, automation, cli]
published: true
---

GitHub: <https://github.com/jiunbae/settings>

새 컴퓨터 사면 가장 먼저 하는 일이 dotfile 세팅이다. git clone하고 스크립트 돌리고, 이런 저런 설정 파일 심링크 걸고... 어쩌다 가끔 있는 일이지만 몇 번 반복하다 보면 한번에 모두 설치하고 싶은 생각이 들어 이전에는 dotfiles를 모두 복사하는 식으로 관리했었다.

최근에 agents를 활용해서 개선하고 싶은 부분들을 개선했다. 크게 macOS와 linux (wsl) 등 환경에 따른 특수한 부분들을 처리하고 모듈 구조로 나눠 필요한것만 설치하고 확장성을 고려했다. 또한 너무 무겁던 쉘을 간소화 하고 빠르게 로딩될 수 있도록 최소한의 플러그인들만 설치하고 지연 로딩도 적용했다.

이 설정은 나를 위한 개인화 설정이므로, 사용성에 맞게 수정해서 자신만의 레포로 관리하는 편이 용이할 것이다.

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

| 구성 요소 | 설명 | 링크 |
|-----------|------|------|
| **zsh** | Zsh + zinit 플러그인 매니저 + Powerlevel10k 프롬프트 | [zsh](https://www.zsh.org/) · [zinit](https://github.com/zdharma-continuum/zinit) · [Powerlevel10k](https://github.com/romkatv/powerlevel10k) |
| **nvim** | NeoVim + LazyVim (modular Neovim setup) | [NeoVim](https://neovim.io/) · [LazyVim](https://github.com/LazyVim/LazyVim) |
| **tmux** | tmux + TPM (터미널 멀티플렉서 + 플러그인 매니저) | [tmux](https://github.com/tmux/tmux) · [TPM](https://github.com/tmux-plugins/tpm) |
| **rust** | Rust toolchain + cargo-binstall (바이너리 패키지 인스톨러) | [Rust](https://www.rust-lang.org/) · [cargo-binstall](https://github.com/cargo-bins/cargo-binstall) |
| **uv** | 빠른 Python 패키지 매니저 (pip보다 10-100배 빠름) | [uv](https://github.com/astral-sh/uv) |
| **tools** | eza, fd, bat, ripgrep, fzf | - |
| **tools-extra** | delta, dust, procs, bottom | - |
| **hishtory** | 크로스 디바이스 히스토리 동기화 | [hishtory](https://github.com/ddworken/hishtory) |

## 도구들

대체할 도구들:

**`eza` (ls 대체)**

아이콘과 git 통합이 들어간 ls다. 파일 권한, 크기, 수정일 한눈에 볼 수 있고, git 상태도 색깔로 표시해준다. 어떤 파일이 수정됐는지, 새로 추가됐는지 바로 알 수 있어서 git 작업할 때 편하다.

[GitHub: eza-community/eza](https://github.com/eza-community/eza)

**`bat` (cat 대체)**

문법 하이라이팅이 들어간 cat이다. 소스 코드를 볼 때 가독성이 확 달라진다. 줄 번호도 나오고, 터미널에서 diff 보듯이 파일 내용을 편하게 볼 수 있다. 로그 파일이나 설정 파일 볼 때도 syntax highlighting 덕분에 훨씬 읽기 쉽다.

[GitHub: sharkdp/bat](https://github.com/sharkdp/bat)

**`fd` (find 대체)**

간단하고 빠른 find다. 재귀 검색, 정규식 지원, 확장자 필터 등 다 있지만 명령어가 훨씬 직관적이다. `find . -name "*.ts"` 대신 `fd "\.ts$"`로 충분하다. git 디렉토리는 자동으로 무시해서 실제로 원하는 파일만 빠르게 찾아준다. 프로젝트 뒤지면서 파일 찾을 때 필수다.

[GitHub: sharkdp/fd](https://github.com/sharkdp/fd)

**`ripgrep` (grep 대체)**

정규식 검색이 엄청나게 빠르다. rust로 짜여서 그런지 몇십만 줄 코드에서도 순식간에 결과가 나온다. 코드베이스에서 특정 함수나 변수 찾을 때, 로그에서 에러 패턴 찾을 때, 설정 파일에서 특정 옵션 검색할 때 다 쓴다. `-t` 플래그로 파일 타입(`py`, `ts`, `js` 등) 필터링도 되어서 실무에서 딱 좋다.

[GitHub: BurntSushi/ripgrep](https://github.com/BurntSushi/ripgrep)

**`delta` (git diff 대체)**

git diff 훨씬 보기 좋게 해준다. 줄 단위 diff가 아닌 단어 단위 diff, 문법 하이라이팅, side-by-side 뷰 등 다 제공한다. 코드 리뷰할 때나 PR 확인할 때 직관적으로 무엇이 바뀌었는지 한눈에 보인다.

[GitHub: dandavison/delta](https://github.com/dandavison/delta)

**`dust` (du 대체)**

디스크 사용량 보여주는 건데 트리 구조로 보여준다. 어느 디렉토리가 용량을 많이 잡고 있는지 한눈에 파악할 수 있다. 서버나 노트북 디스크 꽉 찼을 때, 빠르게 공간 낭비되는 곳 찾을 때 유용하다.

[GitHub: bootandy/dust](https://github.com/bootandy/dust)

**`procs` (ps 대체)**

프로세스 정보를 보기 좋게 보여준다. ps 명령어는 출력이 난해한데 procs는 테이블 형태로 깔끔하다. CPU, 메모리, 실행 시간 다 한눈에 볼 수 있고, 검색/필터링도 쉽다.

[GitHub: dalance/procs](https://github.com/dalance/procs)

**`bottom` (htop 대체)**

현대적 UI의 시스템 모니터다. htop보다 더 직관적이고, GPU, 디스크 I/O, 네트워크 등 더 많은 정보를 보여준다. 커맨드라인에서 서버 상태 모니터링할 때 쓴다.

[GitHub: ClementTsang/bottom](https://github.com/ClementTsang/bottom)

**`fzf` (fuzzy finder)**

거의 모든 것에 연결할 수 있는 fuzzy finder다. `Ctrl+R`로 히스토리 검색, 파일 검색, 프로세스 죽이기 등 다 가능하다. 터미널에서 무언가 찾을 때 거의 표준처럼 쓰게 된다.

[GitHub: junegunn/fzf](https://github.com/junegunn/fzf)

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
