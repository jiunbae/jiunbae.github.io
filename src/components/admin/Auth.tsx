/**
 * GitHub 인증 컴포넌트
 * Personal Access Token 입력 및 검증
 */

import React, { useState } from 'react';
import { useGitHub } from '@/contexts/GitHubContext';

const Auth: React.FC = () => {
  const { login, logout, isAuthenticated, isValidating } = useGitHub();
  const [tokenInput, setTokenInput] = useState('');
  const [error, setError] = useState('');
  const [showToken, setShowToken] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!tokenInput.trim()) {
      setError('토큰을 입력해주세요.');
      return;
    }

    const success = await login(tokenInput);

    if (!success) {
      setError('유효하지 않은 토큰입니다. 다시 확인해주세요.');
    } else {
      setTokenInput('');
    }
  };

  const handleLogout = () => {
    if (confirm('로그아웃하시겠습니까? 토큰 정보가 삭제됩니다.')) {
      logout();
    }
  };

  if (isValidating) {
    return (
      <div className="auth-container">
        <div className="auth-loading">
          <p>인증 확인 중...</p>
        </div>
      </div>
    );
  }

  if (isAuthenticated) {
    return (
      <div className="auth-status">
        <span className="auth-status-badge">✓ 인증됨</span>
        <button onClick={handleLogout} className="btn-logout">
          로그아웃
        </button>
      </div>
    );
  }

  return (
    <div className="auth-container">
      <div className="auth-card">
        <h2>GitHub 인증</h2>
        <p className="auth-description">
          블로그를 관리하려면 GitHub Personal Access Token이 필요합니다.
        </p>

        <form onSubmit={handleLogin}>
          <div className="form-group">
            <label htmlFor="token">Personal Access Token</label>
            <div className="token-input-wrapper">
              <input
                id="token"
                type={showToken ? 'text' : 'password'}
                value={tokenInput}
                onChange={(e) => setTokenInput(e.target.value)}
                placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
                className="token-input"
                autoComplete="off"
              />
              <button
                type="button"
                onClick={() => setShowToken(!showToken)}
                className="btn-toggle-visibility"
              >
                {showToken ? '숨기기' : '보기'}
              </button>
            </div>
            {error && <p className="error-message">{error}</p>}
          </div>

          <button type="submit" className="btn-login" disabled={isValidating}>
            {isValidating ? '확인 중...' : '로그인'}
          </button>
        </form>

        <div className="auth-help">
          <h3>토큰 생성 방법</h3>
          <ol>
            <li>
              <a
                href="https://github.com/settings/tokens/new"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub Settings → Developer settings → Personal access tokens
              </a>
            </li>
            <li>
              "Generate new token (classic)" 클릭
            </li>
            <li>
              권한 설정: <code>repo</code> 체크
            </li>
            <li>
              생성된 토큰 복사하여 위 입력란에 붙여넣기
            </li>
          </ol>
          <p className="auth-warning">
            ⚠️ 보안 주의: 토큰은 브라우저에 저장됩니다. 공용 컴퓨터에서 사용하지 마세요.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Auth;
