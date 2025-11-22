/**
 * 블로그 관리자 페이지
 * /admin 경로로 접근
 */

import React, { useState } from 'react';
import { GitHubProvider, useGitHub } from '@/contexts/GitHubContext';
import Auth from '@/components/admin/Auth';
import PostList from '@/components/admin/PostList';
import Editor from '@/components/admin/Editor';
import { Post } from '@/utils/github';
import '@/styles/admin.scss';

const AdminContent: React.FC = () => {
  const { isAuthenticated } = useGitHub();
  const [selectedPost, setSelectedPost] = useState<Post | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [postType, setPostType] = useState<'post' | 'note'>('post');

  const handleSelectPost = (post: Post | null) => {
    setSelectedPost(post);
    if (post) {
      setIsEditing(true);
      // post.path에서 타입 추론
      setPostType(post.path.includes('/posts/') ? 'post' : 'note');
    }
  };

  const handleNewPost = (type: 'post' | 'note') => {
    setSelectedPost(null);
    setPostType(type);
    setIsEditing(true);
  };

  const handleSaved = () => {
    setIsEditing(false);
    setSelectedPost(null);
  };

  const handleCancel = () => {
    if (confirm('작성 중인 내용이 임시 저장됩니다. 취소하시겠습니까?')) {
      setIsEditing(false);
      setSelectedPost(null);
    }
  };

  if (!isAuthenticated) {
    return (
      <div className="admin-page">
        <div className="admin-container">
          <Auth />
        </div>
      </div>
    );
  }

  return (
    <div className="admin-page">
      <header className="admin-header">
        <h1>블로그 관리자</h1>
        <Auth />
      </header>

      <div className="admin-container">
        {isEditing ? (
          <Editor
            post={selectedPost}
            postType={postType}
            onSaved={handleSaved}
            onCancel={handleCancel}
          />
        ) : (
          <PostList
            onSelectPost={handleSelectPost}
            onNewPost={handleNewPost}
            selectedPost={selectedPost}
          />
        )}
      </div>
    </div>
  );
};

const AdminPage: React.FC = () => {
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
