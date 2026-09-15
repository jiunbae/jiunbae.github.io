/**
 * Which service card on /status/ corresponds to which Kubernetes namespace.
 *
 * Kept beside the service list rather than inside it: a namespace is a
 * deployment detail that only the status page cares about, and most entries in
 * `services.ts` have none — they are apps, extensions or sites that never ran
 * as a namespace.
 *
 * This map must stay a subset of what `GET /public/status` is willing to
 * publish. That endpoint has its own allowlist, so a title added here without
 * the matching namespace on the API side simply stays on its build-time state
 * instead of leaking anything.
 */
export const STATUS_NAMESPACE_BY_TITLE: Record<string, string> = {
  Bubbles: 'bubbles',
  Crux: 'crux',
  Daily: 'daily',
  '문짓': 'munjit',
  '사이': 'sai',
  '관계 인벤토리': 'relation-inventory',
  '도믿맨: 퇴근길 생존기': 'domidman',
  Chartlog: 'chartlog',
  Finchi: 'finchi',
  Flatten: 'flatten',
  IssueBoard: 'issueboard',
  Kongbu: 'kongbu',
  Nolbul: 'nolbul',
  'Nova Pouch': 'nova-pouch',
  'Oh My Prompt': 'oh-my-prompt',
  Ssudam: 'ssudam',
  Tokka: 'tokka',
};
