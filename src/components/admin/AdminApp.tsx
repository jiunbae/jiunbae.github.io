import React, { useState, useCallback } from 'react';
import { GitHubProvider, useGitHub } from './context/GitHubContext';
import AuthView from './views/AuthView';
import ListView from './views/ListView';
import EditorView from './views/EditorView';
import { parseContentPath, type ContentType } from './lib/content-paths';

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
      <span style={{ marginRight: 'auto', color: 'var(--gray-3)' }}>
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
    <GitHubProvider>
      <div className="admin-container">
        <AdminRouter />
      </div>
    </GitHubProvider>
  );
}
