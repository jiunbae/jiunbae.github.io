# Cold Review Round Archive (R11–R18)

8 라운드의 4-reviewer cold review 결과. 각 라운드 노트는 reviewer별 점수,
top strengths/weaknesses, 적용된 fix, build/audit metric을 포함합니다.

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
