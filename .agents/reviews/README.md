# Cold Review Round Archive (R11–R18)

8 라운드의 4-reviewer cold review 결과. 각 라운드 노트는 reviewer별 점수,
top strengths/weaknesses, 적용된 fix, build/audit metric을 포함합니다.

## 왜 R11부터인가 (Phase 1, R1–R10 자료 부재)

`R11`은 임의의 시작점이 아니라 **Phase 2 (블로그 전체 적용) baseline**입니다.

이전 단계인 **Phase 1** — 디자인 시스템 페이지(`/design`) 자체를 구축한
단계 — 도 4-reviewer cold review로 진행되었지만, 그 시점에는
`score-tracker.mjs`, `.agents/personas/`, 라운드별 노트 작성 인프라가
**존재하지 않았습니다**. 점수는 conversation 안에만 존재했고 파일로
누적되지 않았기 때문에 **R1–R10의 라운드별 자료는 보존되지 않았습니다**.

Phase 1에 대해 남아있는 흔적은 다음과 같습니다:

- `src/pages/design/changelog.astro` v0.1 entry — Phase 1 산출물 요약
- commit `a557b92` (또는 그 근방) — Phase 1 종료 시점 스냅샷
- "28 라운드 cold review, 평균 9.557" 같은 conversation 메타 정보

이 아카이브는 **append-only ground truth**라는 정책에 따라, 후술
재구성으로 R1–R10을 만들지 않습니다 (점수 cherry-pick 금지). R11이
보존 가능했던 첫 라운드이므로 그 지점을 Phase 2의 baseline으로
기록합니다.

## 점수 추이

| Round | A (DS) | B (UX/한글) | C (a11y) | D (IA/Grid) | Avg | Δ avg | Notes |
|---|---|---|---|---|---|---|---|
| R11 | 9.3 | 9.2 | 9.5 | 7.2 | **8.80** | — | baseline |
| R12 | 9.4 | 9.3 | 9.6 | 8.6 | **9.225** | +0.42 | container scale 통합 + 초기 sweep |
| R13 | 9.5 | 9.4 | 9.7 | 9.1 | **9.425** | +0.20 | spacing/motion/radius sweep |
| R14 | 9.6 | 9.5 | 9.8 | 9.1 | **9.488** | +0.06 | SVG aria-hidden, prose radius cleanup |
| R15 | 9.6 | 9.5 | 9.6 | 9.2 | **9.475** | −0.01 | site-wide sweep; converter contrast regression |
| R16 | 9.6 | 9.6 | 9.7 | 9.3 | **9.55** | +0.08 | admin.scss tokenized, contrast fixes |
| R17 | 9.7 | 9.65 | 9.7 | 9.4 | **9.613** | +0.06 | stylelint 0, admin/contact/about cleanups |
| R18 | 9.8 | 9.68 | 9.7 | 9.5 | **9.67** | +0.06 | axe + visual pipeline; 0 a11y violations |

**총 Δ R11→R18: +0.87** (8.80 → 9.67)

## 차원별 변화

| Dim | R11 | R18 | Δ |
|---|---|---|---|
| A — DS Adoption | 9.3 | 9.8 | +0.5 |
| B — UX / 한글 | 9.2 | 9.68 | +0.48 |
| C — Accessibility | 9.5 | 9.7 | +0.2 |
| D — IA / Gestalt | 7.2 | 9.5 | **+2.3** |

D 차원이 가장 큰 개선 폭. 초기 점수가 낮았던 이유: container scale 중복,
mixin 미적용, h1 전략 불일치, raw spacing pocket. R12 한 라운드에서
+1.4 점프 (container 통합 + spacing sweep).

## 자동화 metric 추이

| Round | Token adoption | axe violations | stylelint errors |
|---|---|---|---|
| R11–14 | — (수동 grep) | — | — |
| R15 | 75.64% | — | 75 |
| R16 | 85.27% | — | 5 |
| R17 | 91.70% | — | 0 |
| R18 | 91.81% | 173 → 0 | 0 |

`scripts/audit-history.jsonl`이 commit별 ground truth.

## 핵심 변곡점

- **R12 → R13**: container scale 통합 (xs..2xl), 177곳 spacing 토큰화
- **R14 → R15**: views/.module.scss sweep으로 adoption +9.6pp; 그러나
  converter 안에서 `#e53e3e`/undefined `--gray-7` contrast 회귀
- **R15 → R16**: 회귀 즉시 수정 + admin.scss 토큰화로 회복 + 9.55 도달
- **R17 → R18**: Playwright + axe-core pipeline 도입, 173건 자동 발견 →
  같은 라운드에서 모두 해결, 0 violations 달성

## 점수 정책

- reviewer agent의 자기 평가 그대로 기록 (cherry-pick 금지)
- raw 점수 + 평균은 `scripts/scores.json`에 누적 (`npm run score:show`로 출력)
- 라운드 노트는 append-only — 과거 라운드 수정하지 않음
- fix가 적용된 후의 score는 *다음* 라운드에서 측정
