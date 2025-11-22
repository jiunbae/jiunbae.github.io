/**
 * 포스트 목록 컴포넌트
 * 기존 포스트/노트 조회 및 선택
 */

import React, { useState, useEffect } from 'react';
import { useGitHub } from '@/contexts/GitHubContext';
import { getPosts, Post } from '@/utils/github';
import { format } from 'date-fns';

interface PostListProps {
  onSelectPost: (post: Post | null) => void;
  onNewPost: (type: 'post' | 'note') => void;
  selectedPost: Post | null;
}

const PostList: React.FC<PostListProps> = ({ onSelectPost, onNewPost, selectedPost }) => {
  const { octokit } = useGitHub();
  const [posts, setPosts] = useState<Post[]>([]);
  const [notes, setNotes] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'posts' | 'notes'>('posts');

  useEffect(() => {
    loadPosts();
  }, [octokit]);

  const loadPosts = async () => {
    if (!octokit) return;

    setLoading(true);
    try {
      const [postsData, notesData] = await Promise.all([
        getPosts(octokit, 'posts'),
        getPosts(octokit, 'notes'),
      ]);

      setPosts(postsData);
      setNotes(notesData);
    } catch (error) {
      console.error('Failed to load posts:', error);
    } finally {
      setLoading(false);
    }
  };

  const currentList = activeTab === 'posts' ? posts : notes;

  return (
    <div className="post-list">
      <div className="post-list-header">
        <div className="post-list-tabs">
          <button
            className={`tab ${activeTab === 'posts' ? 'active' : ''}`}
            onClick={() => setActiveTab('posts')}
          >
            포스트 ({posts.length})
          </button>
          <button
            className={`tab ${activeTab === 'notes' ? 'active' : ''}`}
            onClick={() => setActiveTab('notes')}
          >
            노트 ({notes.length})
          </button>
        </div>

        <div className="post-list-actions">
          <button onClick={() => onNewPost(activeTab === 'posts' ? 'post' : 'note')} className="btn-new">
            + 새 {activeTab === 'posts' ? '포스트' : '노트'}
          </button>
          <button onClick={loadPosts} className="btn-refresh" disabled={loading}>
            새로고침
          </button>
        </div>
      </div>

      <div className="post-list-content">
        {loading ? (
          <div className="post-list-loading">
            <p>로딩 중...</p>
          </div>
        ) : currentList.length === 0 ? (
          <div className="post-list-empty">
            <p>아직 {activeTab === 'posts' ? '포스트' : '노트'}가 없습니다.</p>
            <button onClick={() => onNewPost(activeTab === 'posts' ? 'post' : 'note')} className="btn-new-large">
              첫 {activeTab === 'posts' ? '포스트' : '노트'} 작성하기
            </button>
          </div>
        ) : (
          <ul className="post-list-items">
            {currentList.map((post) => (
              <li
                key={post.path}
                className={`post-list-item ${
                  selectedPost?.path === post.path ? 'active' : ''
                }`}
                onClick={() => onSelectPost(post)}
              >
                <div className="post-item-content">
                  <h3 className="post-item-title">{post.title || '(제목 없음)'}</h3>
                  <p className="post-item-description">
                    {post.description || '(설명 없음)'}
                  </p>
                  <div className="post-item-meta">
                    <span className="post-item-date">
                      {format(new Date(post.date), 'yyyy-MM-dd')}
                    </span>
                    {post.tags.length > 0 && (
                      <span className="post-item-tags">
                        {post.tags.map((tag: string) => (
                          <span key={tag} className="tag">
                            {tag}
                          </span>
                        ))}
                      </span>
                    )}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default PostList;
