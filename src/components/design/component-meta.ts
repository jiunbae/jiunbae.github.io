/*
 * 디자인 시스템 컴포넌트 페이지 데이터 — Button / Chip / Card / Note.
 * [name].astro god route에서 분리. 새 컴포넌트 추가 시 이 파일만 갱신.
 */

export type ComponentStatus = 'Stable' | 'Beta' | 'Deprecated';
export type ComponentName = 'button' | 'chip' | 'card' | 'note';

export const isComponentName = (n: string | undefined): n is ComponentName =>
  n === 'button' || n === 'chip' || n === 'card' || n === 'note';

export interface ComponentMeta {
  title: string;
  status: ComponentStatus;
  purpose: string;
  why: string[];
  anatomy: string[];
  props: { name: string; type: string; default?: string; desc: string }[];
  a11y: string[];
  when: string[];
  whenNot: string[];
  related: { name: string; href: string; reason: string }[];
  importPath: string;
}

export const componentMeta: Record<ComponentName, ComponentMeta> = {
  button: {
    title: 'Button',
    status: 'Stable',
    purpose: '클릭으로 동작을 일으키는 가장 기본적인 인터랙션 단위.',
    why: [
      'Variant는 3개로 한정했습니다. Material의 5개 이상에서 의사결정 비용을 의식적으로 줄였습니다.',
      'Size도 sm/md/lg 세 단계. 흔한 4단계에서 한 단계 줄였습니다. 대부분 md로 끝납니다.',
      'Disabled는 opacity 곱이 아니라 별도 토큰을 씁니다. 곱셈은 대비비를 깨뜨립니다.',
      'Focus ring은 컴포넌트 안에 들어있습니다. 외부 페이지 스타일에 의존하지 않습니다.',
    ],
    anatomy: [
      'Container — radius-md, border 1px, padding-sm/md/lg',
      'Label — font-sans 600 weight (가운데 정렬)',
      'Focus ring — :focus-visible 시 shadow-focus-ring',
    ],
    props: [
      { name: 'variant', type: '"primary" | "secondary" | "ghost"', default: 'primary', desc: 'accent 배경 / 보더만 / hover-only' },
      { name: 'size', type: '"sm" | "md" | "lg"', default: 'md', desc: '패딩과 폰트 크기' },
      { name: 'disabled', type: 'boolean', default: 'false', desc: 'surface-subtle bg + text-disabled (대비 5.7:1 유지)' },
      { name: 'type', type: 'string', default: '"button"', desc: '기본 button, 폼 안에서 submit 등' },
    ],
    a11y: [
      '키보드 Tab으로 도달 가능, Enter/Space로 실행',
      ':focus-visible 시 3px primary 30% alpha ring',
      '아이콘 전용 버튼은 aria-label 필수',
      'disabled 상태에서도 텍스트 대비 5.7:1 유지 (opacity 곱이 아니라 별도 토큰 사용)',
    ],
    when: [
      '명확한 액션 (저장, 보내기, 닫기)',
      '폼 제출',
      '주요 CTA',
    ],
    whenNot: [
      '단순 페이지 이동 — Link 사용',
      '토글 상태 표시 — Chip variant="active" 사용',
      '6개 이상의 옵션 — Dropdown 검토',
    ],
    related: [
      { name: 'Chip', href: '/design/components/chip', reason: '클릭 가능한 작은 단위 (as="button")' },
      { name: 'Card', href: '/design/components/card', reason: '카드 안에서 액션이 필요한 경우' },
    ],
    importPath: "import Button from '@/components/Button.astro';",
  },
  chip: {
    title: 'Chip',
    status: 'Stable',
    purpose: '태그·필터·뱃지 등 작은 상태/카테고리 표시 단위.',
    why: [
      'Span과 button 두 역할을 한 컴포넌트로 묶었습니다. 같은 시각, 다른 의미.',
      'Variant 3개. 색 폭증을 막기 위해 active/default/muted만 둡니다.',
      'Radius는 가장 작은 sm. Pill보다 정보 밀도가 우선입니다.',
      'Astro is:global을 씁니다. JS가 동적 생성하는 +N 뱃지에도 같은 스타일이 갑니다.',
    ],
    anatomy: [
      'Container — radius-sm, border 1px',
      'Label — 13px, 600 weight, 단일 행',
      'Optional: as="button"인 경우 cursor pointer + hover/focus 상태',
    ],
    props: [
      { name: 'variant', type: '"default" | "active" | "muted"', default: 'default', desc: '회색 / primary 강조 / 더 흐림' },
      { name: 'as', type: '"span" | "button"', default: 'span', desc: 'span: 표시용 / button: 클릭 가능' },
    ],
    a11y: [
      'as="button"일 때만 키보드 탐색 가능',
      '필터 chip에는 aria-pressed로 토글 상태 표시',
      '단순 라벨이면 span 유지 (스크린리더 노이즈 최소화)',
    ],
    when: [
      '글의 카테고리/태그 표시',
      '필터 토글',
      '+N 형태의 보조 카운터',
    ],
    whenNot: [
      '주요 액션 — Button',
      '본문 강조 — strong/em',
      '여러 줄에 걸친 라벨',
    ],
    related: [
      { name: 'Button', href: '/design/components/button', reason: '주요 액션을 위한 더 강한 인터랙션 단위' },
      { name: 'Card', href: '/design/components/card', reason: '카드 footer에서 태그로 자주 함께 쓰임' },
    ],
    importPath: "import Chip from '@/components/Chip.astro';",
  },
  card: {
    title: 'Post Card',
    status: 'Stable',
    purpose: '리스트 페이지에서 글/노트 항목을 표시하는 단위.',
    why: [
      'HeadingLevel을 prop으로 받습니다. 페이지마다 outline 위계가 다릅니다.',
      'Tags는 slot입니다. 카드가 칩의 종류를 강제하지 않습니다.',
      'Shadow와 raised surface로 카드가 떠 있음을 보입니다.',
      '본문 max-width 안에서 너비가 가변합니다. 1200px과 720px 모두에서 자연스럽습니다.',
    ],
    anatomy: [
      'Container — surface-raised, border-default, radius-lg, shadow-md',
      'Title — 20px 700 (headingLevel prop으로 h2~h4)',
      'Description — 14px 500 secondary',
      'Footer — tags slot + date',
    ],
    props: [
      { name: 'title', type: 'string', desc: '카드 제목 (필수)' },
      { name: 'description', type: 'string', desc: '한 줄 요약 (선택)' },
      { name: 'date', type: 'string', desc: '날짜 표시 (선택)' },
      { name: 'headingLevel', type: '"h2" | "h3" | "h4"', default: 'h3', desc: '문서 outline 위계' },
    ],
    a11y: [
      'headingLevel을 페이지 outline에 맞춰 설정',
      '카드 전체가 링크라면 article > a 구조 + aria-labelledby',
      'date는 <time datetime> 사용',
    ],
    when: [
      '글 목록 리스트',
      '관련 글 추천',
      '검색 결과 표시',
    ],
    whenNot: [
      '단일 hero/feature — 별도 컴포넌트 검토',
      '데이터 행 — Table',
      '액션 위주 — Button group',
    ],
    related: [
      { name: 'Chip', href: '/design/components/chip', reason: '카드 footer의 태그 표시' },
      { name: 'Button', href: '/design/components/button', reason: '카드 안 액션이 있을 때' },
    ],
    importPath: "import PostCard from '@/components/PostCard.astro';",
  },
  note: {
    title: 'Note',
    status: 'Beta',
    purpose: '글 안에서 별도 시선을 받는 보조 단락 (info, caution, issue).',
    why: [
      '흔한 좌측 막대 대신 로그 엔트리 스타일을 씁니다. Notion이나 GitHub alerts와 다른 인상.',
      'Variant는 info/caution/issue 셋으로 한정했습니다. Material의 6개 이상에서 의사결정을 줄였습니다.',
      '컬러 dot과 모노 라벨을 함께 둡니다. 컬러블라인드도 텍스트로 의미를 봅니다.',
      '아직 Beta입니다. aside 남용과 본문 흐름 침입 패턴을 더 봐야 합니다.',
    ],
    anatomy: [
      'Container — grid 2열 (label 100px + body)',
      'Mark — 컬러 dot + 모노스페이스 라벨 (대문자)',
      'Body — slot으로 자유 콘텐츠',
    ],
    props: [
      { name: 'variant', type: '"info" | "caution" | "issue"', default: 'info', desc: '컬러: primary / warning / error' },
      { name: 'label', type: 'string', desc: '라벨 오버라이드 (기본: info/caution/issue)' },
    ],
    a11y: [
      '<aside> 사용 (role="note"는 의도적으로 부여하지 않습니다 — SR 지원이 일관적이지 않고, <aside>의 암묵 역할로 충분합니다.)',
      '컬러 외에 텍스트 라벨로 의미 보조 (컬러블라인드 대응)',
      '본문 흐름에서 분리되어야 하므로 <aside> 사용',
    ],
    when: [
      '본문 흐름에서 잠시 떼어 강조할 보조 정보',
      '주의 사항 또는 알려진 문제 명시',
      '비기능적 메모 (TODO, NOTE)',
    ],
    whenNot: [
      '에러 메시지 — Alert/Toast',
      '폼 검증 — Field error',
      '중요한 액션 차단 메시지 — Modal',
    ],
    related: [
      { name: 'Card', href: '/design/components/card', reason: '본문 흐름에서 독립된 정보 묶음' },
      { name: 'Button', href: '/design/components/button', reason: 'Note 안에 액션이 필요할 때' },
    ],
    importPath: "import Note from '@/components/Note.astro';",
  },
};
