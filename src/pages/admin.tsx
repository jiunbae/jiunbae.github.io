/**
 * 블로그 관리자 페이지
 * /admin 경로로 접근
 */

import React, { useState, useEffect } from 'react';
import type { PageProps } from 'gatsby';
import { GitHubProvider, useGitHub } from '@/contexts/GitHubContext';
import Auth from '@/components/admin/Auth';
import PostList from '@/components/admin/PostList';
import Editor from '@/components/admin/Editor';
import { Post, Review } from '@/utils/github';
import { Draft } from '@/utils/storage';
import '@/styles/admin.scss';

const AdminContent: React.FC = () => {
  const { isAuthenticated } = useGitHub();
  const [selectedPost, setSelectedPost] = useState<Post | Review | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [postType, setPostType] = useState<'post' | 'note' | 'review'>('post');
  const [loadedDraft, setLoadedDraft] = useState<Draft | null>(null);

  const handleSelectPost = (post: Post | Review | null) => {
    setSelectedPost(post);
    if (post) {
      setIsEditing(true);
      // post.path에서 타입 추론
      if (post.path.includes('/posts/')) {
        setPostType('post');
      } else if (post.path.includes('/reviews/')) {
        setPostType('review');
      } else {
        setPostType('note');
      }
    }
  };

  const handleNewPost = (type: 'post' | 'note' | 'review') => {
    setSelectedPost(null);
    setPostType(type);
    setIsEditing(true);
  };

  const handleSaved = () => {
    setIsEditing(false);
    setSelectedPost(null);
  };

  const handleCancel = () => {
    setIsEditing(false);
    setSelectedPost(null);
    setLoadedDraft(null);
  };

  const handleLoadDraft = (draft: Draft) => {
    setLoadedDraft(draft);
    setSelectedPost(null);
    setPostType(draft.type);
    setIsEditing(true);
  };

  return (
    <div className="admin-page">
      {!isAuthenticated ? (
        <div className="admin-container">
          <Auth />
        </div>
      ) : (
        <>
          <div className="admin-header-actions">
            <Auth />
          </div>

          <div className="admin-container">
            {isEditing ? (
              <Editor
                post={selectedPost}
                postType={postType}
                loadedDraft={loadedDraft}
                onSaved={handleSaved}
                onCancel={handleCancel}
              />
            ) : (
              <PostList
                onSelectPost={handleSelectPost}
                onNewPost={handleNewPost}
                onLoadDraft={handleLoadDraft}
                selectedPost={selectedPost}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
};

const AdminPage: React.FC<PageProps> = () => {
  // admin 페이지임을 body에 표시
  useEffect(() => {
    document.body.classList.add('admin-page-body');
    return () => {
      document.body.classList.remove('admin-page-body');
    };
  }, []);

  return (
    <GitHubProvider>
      <AdminContent />
    </GitHubProvider>
  );
};

export default AdminPage;

export const Head = () => (
  <>
    <title>관리자 - Blog</title>
    <meta name="robots" content="noindex, nofollow" />
  </>
);
