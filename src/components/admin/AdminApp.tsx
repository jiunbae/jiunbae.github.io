import React, { Component, useState, useCallback, type ReactNode } from 'react';
import { GitHubProvider, useGitHub } from './context/GitHubContext';
import AuthView from './views/AuthView';
import ListView from './views/ListView';
import EditorView from './views/EditorView';
import { parseContentPath, type ContentType } from './lib/content-paths';

class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="auth-container">
          <div className="auth-card">
            <h2>Something went wrong</h2>
            <p className="auth-description">{this.state.error.message}</p>
            <button
              className="btn-login"
              onClick={() => this.setState({ error: null })}
            >
              Try again
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

type View =
  | { name: 'auth' }
  | { name: 'list' }
  | { name: 'editor'; path?: string; contentType: ContentType };

function AdminRouter() {
  const { isAuthenticated, isLoading, user, logout } = useGitHub();
  const [view, setView] = useState<View>({ name: 'list' });

  const handleEdit = useCallback((path: string) => {
    const contentType = parseContentPath(path)?.type ?? 'posts';
    setView({ name: 'editor', path, contentType });
  }, []);

  const handleNew = useCallback((type: ContentType) => {
    setView({ name: 'editor', contentType: type });
  }, []);

  const handleBack = useCallback(() => {
    setView({ name: 'list' });
  }, []);

  if (isLoading) {
    return <div className="auth-loading">Loading...</div>;
  }

  if (!isAuthenticated) {
    return <AuthView />;
  }

  const header = (
    <div className="admin-header-actions">
      <span className="admin-user-login">
        {user?.login}
      </span>
      <button className="btn-logout" onClick={logout}>
        Logout
      </button>
    </div>
  );

  switch (view.name) {
    case 'list':
      return <>{header}<ListView onEdit={handleEdit} onNew={handleNew} /></>;
    case 'editor':
      return (
        <>{header}<EditorView
          path={view.path}
          contentType={view.contentType}
          onBack={handleBack}
        /></>
      );
    default:
      return <>{header}<ListView onEdit={handleEdit} onNew={handleNew} /></>;
  }
}

export default function AdminApp() {
  return (
    <ErrorBoundary>
      <GitHubProvider>
        <div className="admin-container">
          <AdminRouter />
        </div>
      </GitHubProvider>
    </ErrorBoundary>
  );
}
