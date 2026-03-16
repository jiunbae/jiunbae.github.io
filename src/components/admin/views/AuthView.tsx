import { useState } from "react";
import { useGitHub } from "../context/GitHubContext";
import { getOAuthLoginUrl } from "../lib/jiun-api";

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
        <h2>Admin Login</h2>
        <p className="auth-description">
          Sign in with GitHub to manage blog content.
        </p>

        {displayError && (
          <div className="auth-error"><p>{displayError}</p></div>
        )}

        {isLoading ? (
          <div className="auth-loading">Authenticating...</div>
        ) : (
          <>
            <button
              type="button"
              className="btn-login"
              onClick={handleOAuthLogin}
              style={{ marginBottom: "1rem" }}
            >
              Login with GitHub
            </button>

            <div style={{ textAlign: "center", margin: "1rem 0", color: "var(--gray-5)", fontSize: "0.875rem" }}>
              <span>or</span>
            </div>

            {!showPat ? (
              <button
                type="button"
                className="btn-cancel"
                onClick={() => setShowPat(true)}
                style={{ width: "100%" }}
              >
                Use Personal Access Token
              </button>
            ) : (
              <form onSubmit={handlePatSubmit}>
                <div className="form-group">
                  <label htmlFor="token">GitHub Token</label>
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
                  className="btn-save"
                  disabled={isLoading || !token.trim()}
                >
                  {isLoading ? "Logging in..." : "Login with PAT"}
                </button>
              </form>
            )}
          </>
        )}

        <div className="auth-warning">
          All changes will be committed directly to the{" "}
          <strong>main</strong> branch.
        </div>
      </div>
    </div>
  );
}
