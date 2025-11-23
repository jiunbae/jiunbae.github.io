/**
 * 포스트 목록 컴포넌트
 * 기존 포스트/노트 조회 및 선택
 */

import React, { useState, useEffect } from 'react';
import { useGitHub } from '@/contexts/GitHubContext';
import { getPosts, getReviews, Post, Review } from '@/utils/github';
import { format } from 'date-fns';
import DraftManager from './DraftManager';
import { Draft } from '@/utils/storage';

interface PostListProps {
  onSelectPost: (post: Post | Review | null) => void;
  onNewPost: (type: 'post' | 'note' | 'review') => void;
  onLoadDraft: (draft: Draft) => void;
  selectedPost: Post | Review | null;
}

const PostList: React.FC<PostListProps> = ({ onSelectPost, onNewPost, onLoadDraft, selectedPost }) => {
  const { octokit } = useGitHub();
  const [posts, setPosts] = useState<Post[]>([]);
  const [notes, setNotes] = useState<Post[]>([]);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'posts' | 'notes' | 'reviews'>('posts');

  useEffect(() => {
    loadPosts();
  }, [octokit]);

  const loadPosts = async () => {
    if (!octokit) return;

    setLoading(true);
    try {
      const [postsData, notesData, reviewsData] = await Promise.all([
        getPosts(octokit, 'posts'),
        getPosts(octokit, 'notes'),
        getReviews(octokit),
      ]);

      setPosts(postsData);
      setNotes(notesData);
      setReviews(reviewsData);
    } catch (error) {
      console.error('Failed to load posts:', error);
    } finally {
      setLoading(false);
    }
  };

  const currentList = activeTab === 'posts' ? posts : activeTab === 'notes' ? notes : reviews;

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
          <button
            className={`tab ${activeTab === 'reviews' ? 'active' : ''}`}
            onClick={() => setActiveTab('reviews')}
          >
            리뷰 ({reviews.length})
          </button>
        </div>

        <div className="post-list-actions">
          <DraftManager onLoadDraft={onLoadDraft} currentDraftId="" />
          <button
            onClick={() =>
              onNewPost(activeTab === 'posts' ? 'post' : activeTab === 'notes' ? 'note' : 'review')
            }
            className="btn-new"
          >
            + 새 {activeTab === 'posts' ? '포스트' : activeTab === 'notes' ? '노트' : '리뷰'}
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
            <p>
              아직{' '}
              {activeTab === 'posts' ? '포스트' : activeTab === 'notes' ? '노트' : '리뷰'}가
              없습니다.
            </p>
            <button
              onClick={() =>
                onNewPost(
                  activeTab === 'posts' ? 'post' : activeTab === 'notes' ? 'note' : 'review'
                )
              }
              className="btn-new-large"
            >
              첫 {activeTab === 'posts' ? '포스트' : activeTab === 'notes' ? '노트' : '리뷰'}{' '}
              작성하기
            </button>
          </div>
        ) : (
          <ul className="post-list-items">
            {currentList.map((item) => {
              const isReview = 'mediaType' in item;
              const post = item as Post;
              const review = item as Review;

              return (
                <li
                  key={item.path}
                  className={`post-list-item ${
                    selectedPost?.path === item.path ? 'active' : ''
                  }`}
                  onClick={() => onSelectPost(item)}
                >
                  <div className="post-item-content">
                    <h3 className="post-item-title">
                      {item.title || '(제목 없음)'}
                      {item.published === false && (
                        <span className="unpublished-badge"> (비공개)</span>
                      )}
                    </h3>
                    <p className="post-item-description">
                      {isReview ? review.oneLiner : post.description || '(설명 없음)'}
                    </p>
                    <div className="post-item-meta">
                      <span className="post-item-date">
                        {format(new Date(item.date), 'yyyy-MM-dd')}
                      </span>
                      {item.tags.length > 0 && (
                        <span className="post-item-tags">
                          {item.tags.map((tag: string) => (
                            <span key={tag} className="tag">
                              {tag}
                            </span>
                          ))}
                        </span>
                      )}
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
};

export default PostList;
