import { useState } from "react";
import { useGitHub } from "../context/GitHubContext";
import { getOAuthLoginUrl } from "../lib/jiun-api";

const GitHubIcon = () => (
  <svg width="20" height="20" viewBox="0 0 98 96" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
    <path fillRule="evenodd" clipRule="evenodd" d="M48.854 0C21.839 0 0 22 0 49.217c0 21.756 13.993 40.172 33.405 46.69 2.427.49 3.316-1.059 3.316-2.362 0-1.141-.08-5.052-.08-9.127-13.59 2.934-16.42-5.867-16.42-5.867-2.184-5.704-5.42-7.17-5.42-7.17-4.448-3.015.324-3.015.324-3.015 4.934.326 7.523 5.052 7.523 5.052 4.367 7.496 11.404 5.378 14.235 4.074.404-3.178 1.699-5.378 3.074-6.6-10.839-1.141-22.243-5.378-22.243-24.283 0-5.378 1.94-9.778 5.014-13.2-.485-1.222-2.184-6.275.486-13.038 0 0 4.125-1.304 13.426 5.052a46.97 46.97 0 0 1 12.214-1.63c4.125 0 8.33.571 12.213 1.63 9.302-6.356 13.427-5.052 13.427-5.052 2.67 6.763.97 11.816.485 13.038 3.155 3.422 5.015 7.822 5.015 13.2 0 18.905-11.404 23.06-22.324 24.283 1.78 1.548 3.316 4.481 3.316 9.126 0 6.6-.08 11.897-.08 13.526 0 1.304.89 2.853 3.316 2.364 19.412-6.52 33.405-24.935 33.405-46.691C97.707 22 75.788 0 48.854 0z" />
  </svg>
);

export default function AuthView() {
  const { loginWithToken, error: contextError, isLoading } = useGitHub();
  const [showPat, setShowPat] = useState(false);
  const [token, setToken] = useState("");
  const [showToken, setShowToken] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const displayError = localError || contextError;

  const handleOAuthLogin = () => {
    window.location.href = getOAuthLoginUrl();
  };

  const handlePatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);
    if (!token.trim()) {
      setLocalError("GitHub token is required");
      return;
    }
    await loginWithToken(token.trim());
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-icon">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
              <path d="M18.375 2.625a1 1 0 0 1 3 3l-9.013 9.014a2 2 0 0 1-.853.505l-2.873.84a.5.5 0 0 1-.62-.62l.84-2.873a2 2 0 0 1 .506-.852z" />
            </svg>
          </div>
          <h2>Blog Admin</h2>
          <p className="auth-description">
            Sign in to create, edit, and manage blog content.
          </p>
        </div>

        {displayError && (
          <div className="auth-error"><p>{displayError}</p></div>
        )}

        {isLoading ? (
          <div className="auth-loading">
            <div className="auth-spinner" />
            <span>Authenticating...</span>
          </div>
        ) : (
          <div className="auth-actions">
            <button
              type="button"
              className="btn-github"
              onClick={handleOAuthLogin}
            >
              <GitHubIcon />
              <span>Continue with GitHub</span>
            </button>

            <div className="auth-divider">
              <span>or</span>
            </div>

            {!showPat ? (
              <button
                type="button"
                className="btn-pat-toggle"
                onClick={() => setShowPat(true)}
              >
                Use Personal Access Token
              </button>
            ) : (
              <form onSubmit={handlePatSubmit} className="pat-form">
                <div className="form-group">
                  <label htmlFor="token">Personal Access Token</label>
                  <div className="token-input-wrapper">
                    <input
                      id="token"
                      type={showToken ? "text" : "password"}
                      value={token}
                      onChange={(e) => setToken(e.target.value)}
                      placeholder="github_pat_..."
                      autoComplete="off"
                    />
                    <button
                      type="button"
                      className="btn-toggle-visibility"
                      onClick={() => setShowToken(!showToken)}
                    >
                      {showToken ? "Hide" : "Show"}
                    </button>
                  </div>
                </div>
                <button
                  type="submit"
                  className="btn-pat-submit"
                  disabled={isLoading || !token.trim()}
                >
                  Sign in
                </button>
              </form>
            )}
          </div>
        )}

        <p className="auth-footer">
          Commits directly to <strong>main</strong> branch
        </p>
      </div>
    </div>
  );
}
