const API_BASE = import.meta.env.PUBLIC_API_BASE ?? "https://api.jiun.dev";

export interface JiunApiUser {
  id: string;
  provider: string;
  username: string;
  displayName: string;
  avatarUrl: string | null;
}

export function getOAuthLoginUrl(): string {
  const redirectUri = `${window.location.origin}/admin`;
  return `${API_BASE}/auth/github-admin?redirect_uri=${encodeURIComponent(redirectUri)}`;
}

export async function exchangeAuthCode(code: string): Promise<string> {
  const res = await fetch(`${API_BASE}/auth/exchange`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ code }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || "Failed to exchange auth code");
  }

  const data = await res.json();
  return data.accessToken;
}

export async function fetchCurrentUser(accessToken: string): Promise<{
  user: JiunApiUser;
  githubToken?: string;
}> {
  const res = await fetch(`${API_BASE}/auth/me`, {
    headers: { Authorization: `Bearer ${accessToken}` },
    credentials: "include",
  });

  if (!res.ok) {
    throw new Error("Failed to fetch user");
  }

  return res.json();
}

export async function refreshAccessToken(): Promise<string | null> {
  try {
    const res = await fetch(`${API_BASE}/auth/refresh`, {
      method: "POST",
      credentials: "include",
    });

    if (!res.ok) return null;

    const data = await res.json();
    return data.accessToken;
  } catch {
    return null;
  }
}

export async function logoutApi(): Promise<void> {
  await fetch(`${API_BASE}/auth/logout`, {
    method: "POST",
    credentials: "include",
  }).catch(() => {});
}
