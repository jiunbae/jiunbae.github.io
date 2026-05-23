/*
 * 디자인 시스템 메타 — Cover / Colophon / Changelog가 공유하는 단일 진실 소스.
 *
 * 이전엔 "Issue №01" 표시 텍스트, 날짜, 버전, 토큰/컴포넌트 개수가
 * 3-4곳에 흩어져 있었습니다. 다음 발행본에서는 이 파일 하나만 갱신하면 됩니다.
 */

export const designSections = [
  { id: 'cv-01', short: 'Color', label: 'Color' },
  { id: 'cv-02', short: 'Type', label: 'Typography' },
  { id: 'cv-03', short: 'Foundation', label: 'Foundation' },
  { id: 'cv-04', short: 'In Use', label: 'In Use' },
  { id: 'cv-05', short: 'A11y', label: 'Accessibility' },
  { id: 'cv-06', short: 'Content', label: 'Content' },
] as const;

export const designComponents = [
  { name: 'Button', href: '/design/components/button' },
  { name: 'Chip', href: '/design/components/chip' },
  { name: 'Card', href: '/design/components/card' },
  { name: 'Note', href: '/design/components/note' },
] as const;

/*
 * Counts:
 *   - componentCount는 designComponents 배열에서 자동 도출 — drift 불가능.
 *   - tokenCount는 _tokens.scss + _global.scss의 CSS custom property 합. 정확한
 *     자동 카운트가 빌드 단계에서 어렵기에 정수로 유지하고 changelog 발행 시 갱신.
 */
export const designMeta = {
  issue: '№01',
  date: '2026.05.22',
  isoDate: '2026-05-22',
  version: 'v0.1',
  tokenCount: 70,
  componentCount: designComponents.length,
} as const;
