---
title: LLM과 함께 개발하기
description: Claude Code로 홈랩 인프라를 구축한 경험
date: 2026-01-04
permalink: /dev-with-llm
tags: [ai, claude, dev]
heroImage:
heroImageAlt:
published: false
---
# LLM과 함께 개발하기: Claude Code 활용기

> 목표: **LLM을 "검색 대체재"가 아닌 "페어 프로그래머"로** 활용하기.
> 결과: Terraform, Ansible, Kubernetes 같은 익숙하지 않은 도구를 빠르게 학습하고 실제 인프라 구축에 적용.

---

## 배경: 왜 LLM을 활용하게 됐나

[Home Lab IaC](/posts/home-lab-iac) 글에서 언급했듯이, Terraform과 Ansible로 인프라를 코드화하는 건 처음이었다. 회사에서 Kubernetes는 써봤지만, Proxmox 프로바이더나 Ansible Galaxy 역할 작성은 경험이 없었다.

예전 같았으면 공식 문서를 읽고, Stack Overflow를 뒤지고, 예제 코드를 복붙하면서 삽질했을 것이다. 물론 그 과정에서 배우는 것도 있지만, 시간이 오래 걸리고 "왜 이렇게 해야 하는지"를 이해하기 어려울 때가 많다.

Claude Code를 쓰면서 이 과정이 많이 달라졌다. 질문하고 답변받고 바로 적용하는 사이클이 빨라졌고, 무엇보다 "맥락을 공유하는 대화"가 가능해졌다.

---

## Claude Code 활용 방식

### 1. 코드베이스 이해

처음 IaC 레포를 만들 때, 디렉토리 구조부터 고민이었다. "Terraform과 Ansible을 어떻게 조합하지?", "모듈 분리는 어떻게 하지?"

```
나: Proxmox 환경에서 Terraform + Ansible 조합으로 IaC를 구성하려고 해.
    LXC 컨테이너 여러 개를 만들고, 각각 다른 역할(gateway, monitoring 등)을
    할당하려는데 디렉토리 구조를 어떻게 잡으면 좋을까?

Claude: [디렉토리 구조 제안과 각 폴더의 역할 설명]
        terraform/modules/로 재사용 가능한 LXC 모듈을 만들고,
        ansible/roles/로 서비스별 역할을 분리하면 좋습니다...
```

단순히 "이렇게 하세요"가 아니라, **왜 그렇게 하는지**를 설명해줘서 이해가 빨랐다.

### 2. 문법/API 학습

Terraform HCL 문법이나 Ansible 모듈 사용법을 검색하는 대신, 바로 물어봤다:

```
나: Proxmox Terraform 프로바이더에서 LXC 컨테이너를 만들 때
    root 비밀번호를 설정하려면 어떻게 해?

Claude: proxmox_virtual_environment_container 리소스의
        initialization 블록 안에 user_account를 사용합니다:

        initialization {
          user_account {
            password = var.root_password
          }
        }
```

공식 문서를 찾아보면 10분 걸릴 걸 30초에 해결. 물론 복잡한 케이스는 문서를 직접 확인하지만, 기본적인 문법은 LLM이 훨씬 빠르다.

### 3. 디버깅 / 트러블슈팅

에러 메시지를 그대로 붙여넣으면 원인과 해결책을 제시해준다:

```
나: Ansible에서 이런 에러가 나:
    "Failed to connect to the host via ssh: Permission denied (publickey)"

Claude: SSH 키 인증 실패입니다. 몇 가지 확인할 점:
        1. ansible.cfg에서 private_key_file 경로 확인
        2. 대상 호스트의 authorized_keys에 공개키 등록 여부
        3. SSH 키 권한이 600인지 확인
        ...
```

Stack Overflow에서 비슷한 질문을 찾는 것보다 **내 상황에 맞는 답변**을 받을 수 있다.

### 4. 코드 리뷰 / 개선

작성한 코드를 붙여넣고 피드백을 요청한다:

```
나: 이 Ansible role을 리뷰해줘. 개선할 점이 있을까?

Claude: 몇 가지 제안:
        1. become: yes를 tasks 레벨이 아닌 playbook 레벨에서 설정
        2. 패키지 설치는 loop 대신 name에 리스트로 전달하면 더 효율적
        3. handlers를 사용해서 서비스 재시작을 조건부로
        ...
```

시니어 개발자에게 코드 리뷰 받는 느낌. 혼자 개발할 때 특히 유용하다.

---

## 실제 사례: Proxmox Terraform 모듈 작성

### 문제 상황

Proxmox에서 LXC 컨테이너를 Terraform으로 생성하려고 했다. 공식 문서가 있지만, 내가 원하는 설정(static IP, SSH 키 주입, 특정 템플릿 사용)을 한 번에 찾기 어려웠다.

### LLM 활용 과정

**1단계: 기본 구조 요청**

```
나: Proxmox bpg/proxmox 프로바이더로 LXC 컨테이너를 만드는
    Terraform 모듈 예시 보여줘. Static IP와 SSH 키 설정 포함.
```

Claude가 기본 뼈대를 제공. 이걸 바탕으로 수정 시작.

**2단계: 세부 조정**

```
나: 이 모듈에서 disk size를 변수로 받고 싶어.
    그리고 network_interface에 firewall 옵션도 추가하고 싶은데.

Claude: [수정된 코드]
        disk 블록의 size 속성에 var.disk_size 적용,
        network_interface에 firewall = var.enable_firewall 추가
```

**3단계: 문제 해결**

```
나: terraform apply 하면 "VM 110 does not exist" 에러가 나.
    근데 Proxmox 웹에서 보면 있는데?

Claude: VMID 충돌일 수 있습니다. 기존에 같은 ID로 생성했다가
        삭제한 경우 상태 불일치가 발생할 수 있어요.
        terraform state rm으로 상태 정리 후 재시도하거나,
        Proxmox에서 해당 VM/LXC를 완전히 삭제 후 다시 apply.
```

이렇게 대화하면서 모듈을 완성했다. 결과물:

```hcl
module "gateway" {
  source      = "../../modules/lxc-base"
  vmid        = 110
  hostname    = "gateway"
  ip_address  = "192.168.32.10"
  cores       = 2
  memory      = 2048
  disk_size   = 8
  # ...
}
```

---

## 효과적인 프롬프팅

### 맥락을 충분히 제공하기

```
# Bad
"Ansible에서 nginx 설치하는 법"

# Good
"Debian 12 LXC 컨테이너에서 Ansible로 nginx를 설치하고,
 리버스 프록시로 설정하려고 해. upstream은 여러 개고
 도메인별로 다른 백엔드로 라우팅해야 해."
```

맥락이 많을수록 정확한 답변이 나온다.

### 단계별로 진행하기

복잡한 작업은 한 번에 요청하지 않고 나눈다:

1. "전체 아키텍처 설계해줘"
2. "이 중에서 첫 번째 컴포넌트 구현해줘"
3. "이 코드를 리뷰하고 개선점 알려줘"
4. "테스트 방법 알려줘"

### 반복 개선

첫 답변이 완벽하지 않아도 괜찮다. 피드백을 주면서 개선:

```
나: 이 방식 말고 systemd unit 파일을 직접 만드는 방식은 어때?
Claude: [대안 제시]

나: 근데 이러면 업데이트할 때 불편할 것 같은데
Claude: 맞습니다. 그러면 ansible의 lineinfile 모듈로
        기존 unit 파일을 수정하는 방식은 어떨까요?
```

---

## 주의할 점

### 1. 맹신하지 않기

LLM이 생성한 코드가 항상 맞지는 않다. 특히:

- **버전 차이**: 오래된 API나 deprecated된 옵션을 제안할 때가 있음
- **환경 차이**: 일반적인 설정이 내 환경에서는 안 맞을 수 있음
- **보안**: 개발 편의를 위한 설정(예: TLS 검증 비활성화)을 프로덕션에 그대로 쓰면 안 됨

**반드시 직접 검증하고 이해한 후에 적용.**

### 2. 학습 목적을 잊지 않기

LLM에게 다 시키면 편하지만, 배우는 게 없다. 나는 이렇게 균형을 맞춤:

- **처음 보는 기술**: LLM으로 빠르게 시작하고, 나중에 공식 문서로 깊이 이해
- **이미 아는 기술**: 직접 작성하고, 막히면 LLM에게 질문
- **복잡한 문제**: LLM 답변을 참고하되, 직접 디버깅하며 원리 파악

### 3. 민감 정보 주의

회사 코드나 API 키, 내부 아키텍처 같은 민감 정보는 주의. 나는 홈랩 프로젝트라 크게 걱정 없었지만, 업무에서는 신중해야 한다.

---

## Claude Code 특화 팁

### 파일 읽기 / 수정

Claude Code는 로컬 파일을 읽고 수정할 수 있다. 이게 웹 ChatGPT와의 큰 차이점:

```
나: ansible/roles/nginx/tasks/main.yml 파일 읽어서
    개선할 점 알려줘

Claude: [파일 읽은 후 구체적인 라인 지적하며 피드백]
```

### 멀티 파일 작업

관련 파일들을 한 번에 다룰 수 있다:

```
나: kubernetes/apps/selectchatgpt/ 폴더의 모든 yaml 파일을
    검토하고 kustomization.yaml이 제대로 참조하고 있는지 확인해줘
```

### Bash 실행

테스트나 확인 작업을 직접 실행할 수 있다:

```
나: terraform plan 실행해서 결과 보여줘
Claude: [명령 실행 후 결과 분석]
```

---

## 생산성 변화

체감상 **2~3배** 정도 빨라진 것 같다. 특히:

| 작업 유형      | 예전                   | LLM 활용              |
| -------------- | ---------------------- | --------------------- |
| 새 기술 학습   | 문서 + 튜토리얼 + 삽질 | 대화로 빠르게 이해    |
| 디버깅         | 검색 + StackOverflow   | 에러 메시지 붙여넣기  |
| 보일러플레이트 | 예제 복붙 + 수정       | 요구사항 설명 후 생성 |
| 코드 리뷰      | 혼자 고민              | 피드백 요청           |

물론 LLM 없이도 할 수 있는 일이다. 하지만 **시간과 에너지를 아끼고 핵심에 집중**할 수 있다는 게 큰 장점이다.

---

## 맺음말

LLM을 "코딩해주는 AI"로만 보면 반쪽만 활용하는 것이다. 나는 **"24시간 대기하는 시니어 개발자"**처럼 쓰려고 한다. 모르는 거 물어보고, 코드 리뷰 받고, 아이디어 논의하고.

물론 한계도 있다. 최신 정보가 없을 수 있고, 도메인 특화 지식은 부족할 수 있다. 하지만 홈랩 인프라 구축 같은 "널리 알려진 기술의 조합"에서는 충분히 강력하다.

앞으로 LLM 없이 개발하는 건 상상하기 어렵다. 검색 엔진이 그랬듯이, LLM도 개발자의 기본 도구가 될 것 같다.

---

## 참고

- [Claude Code](https://claude.ai/claude-code)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)
