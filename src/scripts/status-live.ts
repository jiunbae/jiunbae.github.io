/**
 * Live availability for /status/.
 *
 * The page is built statically on GitHub Pages, deliberately off the cluster it
 * reports on, so it survives an outage that takes the whole homelab down. This
 * script fills in what a static build cannot know: whether each service is up
 * right now.
 *
 * Three states matter and all three are shown:
 *   - live data arrived        → render it, stamp the time
 *   - the API did not answer   → say so, and fall back to the committed snapshot
 *   - the snapshot is old      → say how old, because stale data presented as
 *                                current is the failure mode a status page has
 *                                to avoid above all others
 */

const ENDPOINT = 'https://api.jiun.dev/public/status';
const TIMEOUT_MS = 6000;
/** Past this, a snapshot is described as stale rather than quietly shown as fact. */
const STALE_AFTER_MS = 60 * 60 * 1000;

type LiveState = 'operational' | 'degraded' | 'outage' | 'unknown';

interface LiveService {
  name: string;
  state: LiveState;
  uptime7d: number | null;
}

interface LiveStatus {
  generatedAt: string;
  window: string;
  source: 'prometheus' | 'unavailable';
  services: LiveService[];
}

/** Maps a live state onto the class names the page already styles. */
const STATE_CLASS: Record<LiveState, string> = {
  operational: 'status-operational',
  degraded: 'status-degraded',
  outage: 'status-major-outage',
  unknown: 'status-operational',
};

const STATE_LABEL: Record<LiveState, string> = {
  operational: 'Operational',
  degraded: 'Degraded',
  outage: 'Major Outage',
  unknown: 'Unknown',
};

function relativeAge(ms: number): string {
  const minutes = Math.round(ms / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

function setBanner(text: string, tone: 'warn' | 'error') {
  const banner = document.querySelector<HTMLElement>('[data-status-banner]');
  if (!banner) return;
  // Write into the text span so the leading dot survives.
  const target = banner.querySelector<HTMLElement>('[data-status-banner-text]') ?? banner;
  target.textContent = text;
  banner.dataset.tone = tone;
  banner.hidden = false;
}

function setStamp(text: string) {
  const stamp = document.querySelector<HTMLElement>('[data-status-stamp]');
  if (stamp) stamp.textContent = text;
}

function applyServices(services: LiveService[], windowLabel: string) {
  const byName = new Map(services.map((s) => [s.name, s]));
  for (const card of document.querySelectorAll<HTMLElement>('[data-service-ns]')) {
    const live = byName.get(card.dataset.serviceNs ?? '');
    if (!live) continue;

    // An open incident is an editorial statement about something a probe cannot
    // see, so it outranks the metric. Replicas can be up while the thing is
    // useless, and only a person knows that.
    if (card.dataset.incidentActive === 'true') continue;

    for (const cls of Object.values(STATE_CLASS)) card.classList.remove(cls);
    card.classList.add(STATE_CLASS[live.state]);

    const label = card.querySelector<HTMLElement>('.service-status');
    if (label) label.textContent = STATE_LABEL[live.state];

    if (live.uptime7d !== null) {
      const uptime = card.querySelector<HTMLElement>('[data-service-uptime]');
      if (uptime) {
        uptime.textContent = `${(live.uptime7d * 100).toFixed(2)}% ${windowLabel}`;
        uptime.hidden = false;
      }
    }
  }
}

function readSnapshot(): LiveStatus | null {
  const el = document.querySelector<HTMLScriptElement>('#status-snapshot');
  if (!el?.textContent) return null;
  try {
    return JSON.parse(el.textContent) as LiveStatus;
  } catch {
    return null;
  }
}

function showSnapshotFallback(reason: string) {
  const snapshot = readSnapshot();
  if (!snapshot || snapshot.services.length === 0) {
    setBanner(`${reason} Live status is unavailable and there is no recent snapshot to fall back on.`, 'error');
    setStamp('no data');
    return;
  }
  const age = Date.now() - Date.parse(snapshot.generatedAt);
  applyServices(snapshot.services, snapshot.window);
  const staleness = age > STALE_AFTER_MS ? ' The snapshot is old, so treat it as history rather than the current state.' : '';
  setBanner(`${reason} Showing the last known state from ${relativeAge(age)}.${staleness}`, 'error');
  setStamp(`snapshot · ${relativeAge(age)}`);
}

async function load() {
  let payload: LiveStatus;
  try {
    const response = await fetch(ENDPOINT, { signal: AbortSignal.timeout(TIMEOUT_MS) });
    if (!response.ok) throw new Error(String(response.status));
    payload = (await response.json()) as LiveStatus;
  } catch {
    // Not reaching the API is itself the most useful thing this page can report.
    showSnapshotFallback('Cannot reach the status API.');
    return;
  }

  if (payload.source !== 'prometheus' || payload.services.length === 0) {
    showSnapshotFallback('The status API answered but has no availability data right now.');
    return;
  }

  applyServices(payload.services, payload.window);
  setStamp(`live · checked ${relativeAge(Date.now() - Date.parse(payload.generatedAt))}`);
}

void load();
// A status page left open during an incident should follow it.
setInterval(() => void load(), 60000);
