---
title: "카카오톡 대화 분석 서비스를 하루 만에 배포한 이야기"
description: "카카오톡 대화 내보내기 파일을 분석해서 페르소나, 관계, 통계를 보여주는 서비스 Tokka를 AI 에이전트와 함께 하루 만에 만든 과정을 공유합니다."
date: 2026-03-12
permalink: /tokka-kakaotalk-analysis
tags: [AI, Python, FastAPI, React, Gemini, K8s, SideProject]
published: false
---

# 카카오톡 대화 분석 서비스를 하루 만에 배포한 이야기

## 들어가며

> "ㅋ 몇 개 치는지가 성격이라는 거, 알고 계셨나요?"

카카오톡 대화 내보내기 파일을 넣으면, AI가 대화 참여자의 성격과 관계를 분석해주는 서비스를 만들고 싶었습니다. 아이디어는 단순했지만, "하루 안에 배포까지"라는 목표는 야심찼죠.

결론부터 말하면, [Tokka](https://tokka.jiun.dev)(톡까)는 3월 11일 하루 동안 21개의 커밋과 함께 프로덕션에 배포되었습니다. Codex와 Claude Code를 동시에 사용하면서요. 이 글은 그 하루의 기록입니다.

## 핵심 아이디어

카카오톡의 "대화 내보내기" 기능은 텍스트 파일을 생성합니다. 이 파일에는 날짜, 시간, 발신자, 메시지가 담겨 있죠. 여기서 추출할 수 있는 정보가 생각보다 많습니다:

1. **통계 분석**: 메시지 빈도, 응답 시간, 대화 주도, ㅋ 패턴, 활동 시간대
2. **AI 페르소나 분석**: 성격, 커뮤니케이션 스타일, MBTI 추정, 관심사
3. **관계 분석**: 관계 온도, 공감 포인트, 갈등 패턴, 인사이드 조크
4. **SNS 콘텐츠**: 공유 가능한 카드, 트위터/인스타그램용 텍스트

## 아키텍처 결정: 3분 컷

서비스 구조를 정하는 데 3분이 걸렸습니다. 과장이 아니에요. 이미 운영 중인 인프라가 있었으니까요.

```
[사용자] → tokka.jiun.dev (React SPA)
              ↓
         FastAPI (Python) ← Gemini API
              ↓
         Redis (잡 상태 저장)
              ↓
[배포] GitHub → Gitea → Gitea Actions → registry.jiun.dev → ArgoCD → K8s
```

프론트엔드는 React + TypeScript + Vite + Tailwind. 백엔드는 FastAPI. AI는 Google Gemini. 배포는 기존 GitOps 파이프라인 그대로. 새로 만들 건 파서와 분석 로직뿐입니다.

## 첫 번째 벽: 인코딩 지옥

카카오톡 내보내기 파일의 인코딩은 플랫폼마다 다릅니다:

- iOS/macOS: UTF-8 (BOM 포함)
- Windows: CP949 (EUC-KR)
- CSV 내보내기: 또 다른 형식

아무 생각 없이 `open(f, 'r', encoding='utf-8')`로 열면 Windows 파일에서 바로 `UnicodeDecodeError`가 터집니다.

```python
def _open_with_fallback(filepath: Path) -> IO[str]:
    try:
        f = open(filepath, "r", encoding="utf-8-sig")
        f.read(1024)  # 1KB 프로브
        f.seek(0)
        return f
    except UnicodeDecodeError:
        return open(filepath, "r", encoding="cp949")
```

1KB만 먼저 읽어보고 실패하면 CP949로 폴백합니다. 단순하지만 실전에서 잘 동작합니다. 이 문제를 Codex에게 "Windows의 EUC-KR 텍스트로는 진행이 안 되는 것 같습니다"라고 말했더니 바로 이 패턴을 제안해줬어요.

## 두 번째 벽: 파싱 포맷의 다양성

카카오톡 내보내기 형식이 OS마다 다릅니다:

```
# iOS/macOS 형식
2026. 3. 11. 오후 1:24, 홍길동 : 안녕하세요

# Windows 형식
[홍길동] [오후 1:24] 안녕하세요

# CSV 형식
Date,User,Message
```

정규식 3개로 해결했습니다. 파서(`kakao_parser.py`, 413줄)가 파일을 열 때 첫 몇 줄을 보고 형식을 자동 감지합니다. ZIP 파일도 지원해서, 여러 .txt 파일이 들어있으면 시간순으로 정렬해서 머지합니다.

## 세 번째 벽: Gemini API 레이트 리밋

가장 고통스러운 부분이었습니다. 세션 로그를 보면 이 문제와 4시간을 싸웠어요.

```
09:35 - "gemini-3.1-flash-lite is not found"
11:01 - "No module named 'google.api_core'"
11:30 - "gemini-2.5-flash를 제거하고 flash-lite만 사용하세요"
12:07 - 드디어 배포
```

**문제 1**: 모델 이름이 자주 바뀜. `gemini-3.1-flash-lite`가 `gemini-3.1-flash-lite-preview`로 바뀌었는데, 에러 메시지가 "not found"만 던져서 원인 파악에 시간이 걸렸습니다.

**문제 2**: RPM(Requests Per Minute) 제한. Flash Lite는 500 RPM이지만, 페르소나 분석에서 청크별로 여러 번 호출하면 금방 도달합니다.

**해결**: 3단계 방어를 구축했습니다.

```python
# 1. 호출 간 15초 딜레이
await asyncio.sleep(15)

# 2. 모델 폴백
models = ["gemini-3.1-flash-lite-preview", "gemini-2.5-flash-lite"]

# 3. 지수 백오프 재시도 (30s/60s/90s)
for attempt, delay in enumerate([30, 60, 90]):
    try:
        return await call_gemini(prompt)
    except QuotaError:
        await asyncio.sleep(delay)
```

## AI 분석 파이프라인

통계 분석은 순수 Python으로, 페르소나 분석은 Gemini API로 처리합니다.

### 통계 분석 (stats_analyzer.py)

9가지 분석 항목을 순수 Python으로 계산합니다:

- **ㅋ 패턴 분류**: ㅋ 1개 = 미세먼지, 2-3개 = 보통, 4-7개 = 꽤 웃김, 8개+ = 개웃김
- **대화 주도 분석**: 30분 이상 공백 후 먼저 말한 사람을 카운트
- **응답 시간**: 중앙값과 평균 (이상치 제거)
- **사용 빈도**: 시간대별, 요일별, 월별 분포
- **특이 패턴**: 말줄임표(...), 물결표(~~), 영어 사용 비율

### 페르소나 분석 (persona_analyzer.py)

Gemini를 3-패스 청크 방식으로 호출합니다:

**1패스 — 청크 분석**: 500-800개 메시지씩 나눠서 각 청크의 성격, 관심사, 스타일을 분석합니다.

**2패스 — 통합**: 청크별 분석을 모아서 인물별 프로필(한 줄 소개, MBTI 추정, 핵심 가치, 커뮤니케이션 팁)과 관계 분석(관계 온도 1-100, 갈등 패턴, 인사이드 조크)을 생성합니다.

**3패스 — SNS 콘텐츠**: 공유용 트위터 훅, 인스타 캐러셀, "나의 카카오톡 DNA" 카드를 생성합니다.

## 프론트엔드: 분석 결과를 보여주기

React로 결과 페이지를 구성했습니다:

- **Overview Tab**: 페르소나 카드 (이름, MBTI, 한 줄 소개)
- **Persona Tab**: 상세 프로필 (관심사 맵, 핵심 가치, 숨겨진 면)
- **Stats Tab**: 인터랙티브 차트 (Recharts)
- **Share Tab**: SNS 공유용 카드 (html2canvas로 이미지 내보내기)

Zustand으로 상태 관리, Framer Motion으로 애니메이션, Tailwind v4로 스타일링. 이 선택들은 고민 없이 결정했습니다. 빠르게 만들어야 하니까요.

## 배포 파이프라인

"커밋 푸시해서 배포합시다"가 이날 가장 많이 쓴 프롬프트입니다. Codex 히스토리를 보면 최소 5번은 나옵니다.

```
GitHub push → Gitea mirror sync → Gitea Actions
                                        ↓
                                 Docker build (amd64 + arm64)
                                        ↓
                                 registry.jiun.dev push
                                        ↓
                                 IaC repo 이미지 태그 업데이트
                                        ↓
                                 ArgoCD auto-sync → K8s 롤아웃
```

한 번 셋업해두면 `git push`만으로 배포가 완료됩니다. 3월 11일에 이 파이프라인을 5번 이상 돌렸어요. 빌드 실패, 런타임 에러, 모델 변경 — 매번 고치고 다시 배포했습니다.

## 가장 재미있던 순간들

세션 로그에서 발견한 재미있는 대화들:

**개인정보 걱정**:
> "저희 서비스는 궁극적으로 개인정보를 처리하고 있는데 개인정보 처리에 대해 동의를 받아야할것같습니다. 저는 개인사업자가 별도로 없으며 장난감 프로젝트입니다"

장난감 프로젝트인데 법적 의무를 고민하는 개발자의 성실함(?)이 기록에 남았습니다.

**실시간 디버깅 연쇄**:
> 09:23 - "persona, 관계, SNS 페이지가 아무것도 보이지 않습니다"
> 09:24 - "AI가 실패했다면 실패했다고 나타나야합니다"
> 09:25 - "왜 빈화면이 나왔는지 분석하세요"

배포하고 → 확인하고 → 안 되고 → 고치고. 이 사이클이 2분 간격으로 반복됩니다.

**직설적 피드백**:
> "알아서 좀 제대로 찾아서 문제를 해결하는걸 우선으로 진행하세요"
> "문제의 근본원인을 찾아서 해결해야합니다"

AI 에이전트에게 화가 나서 직설적으로 말하는 순간. 프롬프트도 감정이 담깁니다.

## 숫자로 보는 하루

| 지표 | 수치 |
|------|------|
| 총 커밋 수 | 21개 (Day 1) + 10개 (Day 2) |
| 파서 코드 | 413줄 |
| 통계 분석 코드 | 405줄 |
| AI 분석 코드 | 393줄 → 470줄 (보충 호출 추가) |
| AI 리포트 파서 | 457줄 (그룹채팅 포맷 지원) |
| 프론트엔드 | React + TypeScript |
| Gemini API 키 | 5개 (free, round-robin) + 1개 (paid, fallback) |
| "커밋 푸시" 횟수 | 합산 10회 이상 |
| 최대 테스트 대화량 | 289,712건 (16명 그룹채팅) |

## 배포 후 이야기: 그룹채팅이라는 변수

하루 만에 배포하고 끝이 아니었습니다. 실제 사용자가 16명짜리 그룹채팅 28만 건을 넣는 순간, 예상치 못한 문제들이 터졌습니다.

### AI 리포트 파서의 한계

1:1 대화에서는 Gemini가 프롬프트 형식을 잘 따릅니다. `## 한줄 소개`, `## 성격 프로필` 같은 마크다운 헤딩으로요. 그런데 16명 그룹채팅에서는 Gemini가 자기 판단으로 형식을 바꿔버립니다:

```markdown
# 1:1 형식 (프롬프트대로)
# [홍길동]의 카톡 페르소나
## 한줄 소개
분석적이고 감성적인 IT 전문가.

# 그룹채팅 형식 (Gemini가 임의로 변경)
### 1. 홍길동: "데이터로 말하는 추진력"
*   **한줄 소개**: 냉철한 분석과 빠른 실행력을 겸비한 행동 대장.
*   **커뮤니케이션 스타일**: 효율 지향적.
```

프론트엔드 파서가 `## 한줄 소개` 같은 헤딩만 찾고 있었으니, 그룹채팅 결과는 전부 파싱 실패. 사용자에게는 "통계 기반 페르소나 요약"이라는 fallback 메시지만 보였습니다.

**해결**: 파서에 bold-label 블록(`**Label**: content`) 추출 로직을 추가하고, 헤딩 매치 실패 시 bold-label로 폴백하는 2단계 탐색 구조로 개선했습니다. MBTI 정규식도 `/MBTI\s*추정/i`에서 `/MBTI/i`로 완화하고, 쉼표로 구분된 명대사 파싱도 추가했습니다.

### "핵심 멤버 위주"라는 함정

더 근본적인 문제가 있었습니다. 16명 참여자를 보냈는데 Gemini가 "대표 핵심 멤버 위주"라며 4명만 리포트를 생성한 겁니다. 메시지 4만 건을 보낸 사람이 AI 분석에서 빠져있었죠.

```
73,771 msgs  ✓ AI    안태우
44,845 msgs  ✗ 빠짐   박원      ← 2위인데 빠짐
41,243 msgs  ✓ AI    이종서
35,606 msgs  ✗ 빠짐   배지운    ← 4위인데 빠짐
```

**해결**: 2단계 전략을 도입했습니다.

1. **프롬프트 강화**: "모든 참여자를 빠짐없이 포함하세요"를 명시하고 `max_output_tokens`를 8000 → 16000으로 올림
2. **보충 호출**: 합성 후 리포트에서 빠진 참여자를 감지하고, 1000건 이상 메시지를 보낸 참여자만 골라 추가 Gemini 호출

```python
missing = find_missing_persona_participants(persona_report, chat.participants)
high_activity_missing = [name for name in missing
                         if msg_counts.get(name, 0) >= 1000]
if high_activity_missing:
    supplementary = synthesize_supplementary_personas(
        client, chunk_analyses, stats, high_activity_missing, highlights
    )
    persona_report += "\n\n" + supplementary
```

### API 키 풀링 → Free/Paid 분리

초기에는 5개 무료 API 키를 단순 round-robin으로 돌렸습니다. 하지만 실사용에서 무료 키 전체가 쿼타를 소진하면 분석이 완전히 멈추는 문제가 있었습니다.

현재는 **Free + Paid 2단계 구조**로 개선했습니다:

```python
_FREE_KEYS = _load_free_keys()  # GEMINI_API_KEYS env (쉼표 구분, 5개)
_PAID_KEY = os.getenv("GEMINI_API_KEY_PAID", "")  # 유료 키 1개
_key_cycle = itertools.cycle(_FREE_KEYS)

# Phase 1: 무료 키 round-robin 시도
# Phase 2: 모든 무료 키 쿼타 소진 시 → 유료 키 fallback
```

무료 키가 `ResourceExhausted` (429)를 반환하면 다음 키로 순환하고, 전체 소진 시에만 유료 키를 사용합니다. 유료 키는 K8s SealedSecret으로 관리하며 dev/prod 클러스터에 별도로 sealed됩니다.

## 배포 이후: 안정화 작업

하루 만에 배포했다는 건, 하루 만에 만든 기술 부채도 있다는 뜻입니다. 이후 며칠간 안정화 작업을 진행했습니다.

### 해결된 과제들

- ~~**AI 리포트 파싱**: 그룹채팅 형식 미지원~~ → bold-label 폴백 파서로 해결
- ~~**그룹채팅 일부 참여자 누락**~~ → 보충 호출로 해결
- ~~**동시 분석 시 크래시**~~ → 큐 기반 순차 처리로 전환 (웹/이메일 각각 bounded queue + worker thread)
- ~~**API 키 전체 소진 시 서비스 중단**~~ → Free/Paid 2단계 fallback 구조로 해결
- ~~**Sealed Secrets dev/prod 불일치**~~ → dev/prod 모두 SealedSecret 사용으로 통일
- ~~**이메일 실패 시 어떤 파일인지 모름**~~ → 에러 메일에 파일명 포함 (XSS 방지 처리)
- ~~**공개설정 변경 불가 (내 분석)**~~ → JWT 소유자 인증 시 비밀번호 없이 토글 가능
- ~~**Graceful shutdown 미구현**~~ → non-daemon worker threads + `threading.Event` 시그널 + sentinel 패턴 + SIGTERM 안전망 핸들러로 구현. `terminationGracePeriodSeconds: 600` 설정으로 K8s와 연동. 배포 시 진행 중 분석을 안전하게 완료하거나 에러 처리
- ~~**OG 이미지 성능/보안**~~ → Pillow 기반 동적 OG 이미지 생성, 폰트 캐싱(`lru_cache`), 서버사이드 이미지 캐싱, `is_public` 접근 제어, gradient 최적화
- ~~**데모 페이지 불완전**~~ → 샘플 데이터를 실제 분석 결과 포맷에 맞춰 전면 재작성. 페르소나/관계/SNS 탭 모두 fallback 없이 정상 렌더링

### 아직 남은 과제

- **싱글 팟 문제**: nginx, uvicorn, email_worker가 하나의 팟에서 동작. 하나가 죽으면 전부 죽음
- **레이트 리미터 상태**: 인메모리 레이트 리미터는 팟 재시작 시 초기화됨
- **Prometheus 모니터링 고도화**: LLM 호출 성공/실패/쿼타 메트릭은 수집 중이나, 대시보드와 알럿은 미구성

## 마치며

하루 만에 서비스를 배포한다는 건 5년 전에는 상상하기 어려웠습니다. 하지만 지금은 AI 코딩 에이전트, 기존에 구축해둔 GitOps 인프라, 그리고 Gemini API 같은 도구들이 이를 가능하게 합니다.

물론 "하루 만에"의 이면에는 몇 달에 걸쳐 구축한 K8s 클러스터, GitOps 파이프라인, 도메인 설정 같은 인프라가 있습니다. 서비스 로직은 하루 만에 만들었지만, 그것을 배포하고 운영할 수 있는 기반은 하루 만에 만들어지지 않았죠.

그리고 "배포"가 끝이 아니라는 것도 배웠습니다. 1:1 대화에서는 완벽하게 동작하던 파서가 16명 그룹채팅에서 깨지고, Gemini가 프롬프트를 무시하고 자기 형식으로 응답하고, 핵심 멤버만 골라서 리포트를 쓰는 문제들. 실제 데이터가 들어오는 순간 예상치 못한 엣지 케이스가 쏟아집니다. 서비스를 만드는 것보다 서비스를 살아있게 유지하는 게 더 어렵다는 걸 하루 만에 체감했습니다.

그래도 확실한 건, 아이디어에서 프로덕션까지의 거리가 점점 줄어들고 있다는 점입니다. "이거 만들면 재밌겠다"라는 생각이 들면, 다음 날 아침에 실제로 돌아가는 서비스가 있을 수 있는 시대입니다.

카카오톡 대화 파일이 있으시다면, [tokka.jiun.dev](https://tokka.jiun.dev)에서 직접 분석해보세요. 1:1이든 16명 그룹이든, ㅋ의 개수가 당신의 성격을 말해줄 겁니다.
