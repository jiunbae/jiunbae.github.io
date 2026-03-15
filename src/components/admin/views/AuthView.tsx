import { useState } from "react";
import { useGitHub } from "../context/GitHubContext";

export default function AuthView() {
  const { login, error: contextError, isLoading: contextLoading } = useGitHub();

  const [token, setToken] = useState("");
  const [passphrase, setPassphrase] = useState("");
  const [showToken, setShowToken] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const displayError = localError || contextError;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);

    if (!token.trim()) {
      setLocalError("GitHub token is required");
      return;
    }

    await login(token.trim(), passphrase);
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <h2>Admin Login</h2>
        <p className="auth-description">
          Enter your GitHub Personal Access Token to manage content.
        </p>

        {displayError && (
          <div className="auth-error">{displayError}</div>
        )}

        <form onSubmit={handleSubmit}>
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

          <div className="form-group">
            <label htmlFor="passphrase">Passphrase</label>
            <input
              id="passphrase"
              type="password"
              value={passphrase}
              onChange={(e) => setPassphrase(e.target.value)}
              placeholder="Encryption passphrase (optional)"
              autoComplete="off"
            />
          </div>

          <button
            type="submit"
            className="btn-login"
            disabled={contextLoading || !token.trim()}
          >
            {contextLoading ? "Logging in..." : "Login"}
          </button>
        </form>

        <div className="auth-help">
          <h3>How to create a Fine-grained PAT</h3>
          <ol>
            <li>
              Go to <strong>GitHub Settings</strong> &rarr;{" "}
              <strong>Developer settings</strong> &rarr;{" "}
              <strong>Fine-grained tokens</strong>
            </li>
            <li>
              Select repository:{" "}
              <code>jiunbae/jiunbae.github.io</code>
            </li>
            <li>
              Permission: <strong>Contents</strong> (Read and write)
            </li>
          </ol>
        </div>

        <div className="auth-warning">
          Warning: All changes will be committed directly to the{" "}
          <strong>main</strong> branch.
        </div>
      </div>
    </div>
  );
}
