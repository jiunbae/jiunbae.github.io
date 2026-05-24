# `.agents/` — Cold Review Persona & Round Archive

이 디렉토리는 디자인 시스템 cold review에 사용된 **4명의 reviewer persona**와
**라운드별 평가 결과**를 체계화한 아카이브입니다.

## 구조

```
.agents/
├── personas/         — 4명 reviewer의 framework, scope, output 명세
│   ├── A-ds-adoption.md
│   ├── B-nielsen-rui-readability.md
│   ├── C-wcag-aria.md
│   └── D-gestalt-ia-grid.md
├── reviews/          — 라운드별 결과 (R11–R18)
│   ├── README.md     — 점수 추이 + 라운드별 핵심 변화 요약
│   ├── R11.md
│   ├── R12.md
│   ├── ...
│   └── R18.md
└── README.md         — 이 파일
```

## 사용 방법

새로운 cold review 라운드를 돌릴 때:

1. **persona 선택** — `personas/{A,B,C,D}-*.md`에서 framework·scope·output 형식 참조
2. **agent 호출** — 4개를 병렬로 spawn (subagent_type: general-purpose).
   prompt에 persona 본문을 그대로 inline 또는 reference로 사용
3. **점수 기록** — `npm run score add R<N> <A> <B> <C> <D> "<note>"`로 누적
4. **라운드 노트** — `reviews/R<N>.md`에 reviewer별 score, top strengths/
   weaknesses, applied fixes, build/audit metric (axe, adoption %)을 기록

## 자동화 hook

- `scripts/score-tracker.mjs` — 점수 누적 (`reviews/`의 ground truth는 사람이 작성)
- `scripts/token-audit.mjs` — token adoption % 측정 → 라운드 노트에 인용
- `scripts/axe-run.mjs` — WCAG 자동 검증 → reviewer C가 보조 자료로 사용

## 평가 차원

| 코드 | 차원 | Framework |
|---|---|---|
| **A** | DS Adoption | Atomic Design (Brad Frost) + token adoption % |
| **B** | UX / Korean Readability | Nielsen 10 Heuristics + Refactoring UI + 한글 가독성 |
| **C** | Accessibility | WCAG 2.2 AA + ARIA APG 1.2 |
| **D** | Layout / IA / Gestalt | Gestalt principles + IA (Rosenfeld/Morville) + Grid/Vertical rhythm |

각 차원은 1.0–10.0 (one decimal) 스케일. 평균은 단순 산술 평균.

## 기록 정책

- **persona**는 변경 시 changelog에 명시 (framework 추가/제거, scope 변경)
- **review round**는 시간순 append-only. 과거 라운드는 수정하지 않음
- **점수**는 reviewer agent의 자기 평가 그대로 기록. cherry-pick 금지
