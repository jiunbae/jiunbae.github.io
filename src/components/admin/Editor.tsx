/**
 * 마크다운 에디터 컴포넌트
 * Frontmatter, 에디터, 이미지 업로더, Draft 관리 통합
 */

import React, { useState, useEffect, useCallback } from 'react';
import MDEditor from '@uiw/react-md-editor';
import { useGitHub } from '@/contexts/GitHubContext';
import { Post, createOrUpdateFile, createMarkdownFile } from '@/utils/github';
import { saveDraft, Draft } from '@/utils/storage';
import FrontmatterForm from './FrontmatterForm';
import ImageUploader from './ImageUploader';
import DraftManager from './DraftManager';
import { format } from 'date-fns';

interface EditorProps {
  post: Post | null;
  postType: 'post' | 'note';
  onSaved: () => void;
  onCancel: () => void;
}

const Editor: React.FC<EditorProps> = ({ post, postType, onSaved, onCancel }) => {
  const { octokit } = useGitHub();
  const [content, setContent] = useState('');
  const [frontmatter, setFrontmatter] = useState<Omit<Post, 'content' | 'path' | 'sha'>>({
    title: '',
    description: '',
    date: format(new Date(), 'yyyy-MM-dd'),
    slug: '',
    tags: [],
    heroImage: '',
    heroImageAlt: '',
  });
  const [draftId] = useState(`draft_${Date.now()}`);
  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState('');
  const [showImageUploader, setShowImageUploader] = useState(false);
  const [viewMode, setViewMode] = useState<'edit' | 'preview' | 'split'>('split');

  // 포스트 로드 또는 초기화
  useEffect(() => {
    if (post) {
      setContent(post.content);
      setFrontmatter({
        title: post.title,
        description: post.description,
        date: post.date,
        slug: post.slug,
        tags: post.tags,
        heroImage: post.heroImage,
        heroImageAlt: post.heroImageAlt,
      });
    } else {
      // 새 포스트
      setContent('');
      setFrontmatter({
        title: '',
        description: '',
        date: format(new Date(), 'yyyy-MM-dd'),
        slug: '',
        tags: [],
        heroImage: '',
        heroImageAlt: '',
      });
    }
  }, [post]);

  // Draft 자동 저장 (5초마다)
  useEffect(() => {
    const interval = setInterval(() => {
      if (content || frontmatter.title) {
        const draft: Draft = {
          id: draftId,
          title: frontmatter.title || '(제목 없음)',
          content,
          frontmatter,
          type: postType,
          savedAt: new Date().toISOString(),
        };
        saveDraft(draft);
        setSaveStatus('자동 저장됨');
        setTimeout(() => setSaveStatus(''), 2000);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [content, frontmatter, draftId, postType]);

  // Draft 불러오기
  const handleLoadDraft = useCallback((draft: Draft) => {
    setContent(draft.content);
    setFrontmatter(draft.frontmatter);
  }, []);

  // 저장 및 퍼블리시
  const handleSave = async () => {
    if (!octokit) return;

    // 유효성 검사
    if (!frontmatter.title.trim()) {
      alert('제목을 입력해주세요.');
      return;
    }

    if (!frontmatter.slug.trim()) {
      alert('슬러그를 입력해주세요.');
      return;
    }

    if (!content.trim()) {
      alert('내용을 입력해주세요.');
      return;
    }

    setSaving(true);

    try {
      // 마크다운 파일 생성
      const markdownContent = createMarkdownFile(frontmatter, content);

      // 파일 경로 생성
      let filePath: string;
      if (post) {
        // 기존 포스트 수정
        filePath = post.path;
      } else {
        // 새 포스트
        const datePrefix = frontmatter.date;
        const slugName = frontmatter.slug.replace(/^\//, '');

        if (postType === 'post') {
          filePath = `contents/posts/${datePrefix}-${slugName}/index.md`;
        } else {
          filePath = `contents/notes/${datePrefix}-${slugName}.md`;
        }
      }

      // 커밋 메시지
      const commitMessage = post
        ? `Update ${postType}: ${frontmatter.title}`
        : `Add ${postType}: ${frontmatter.title}`;

      // GitHub에 저장
      const success = await createOrUpdateFile(
        octokit,
        filePath,
        markdownContent,
        commitMessage,
        post?.sha
      );

      if (success) {
        setSaveStatus('저장 완료! GitHub Actions가 배포를 시작합니다.');
        setTimeout(() => {
          onSaved();
        }, 2000);
      } else {
        alert('저장에 실패했습니다. 다시 시도해주세요.');
      }
    } catch (error) {
      console.error('Failed to save post:', error);
      alert('저장 중 오류가 발생했습니다.');
    } finally {
      setSaving(false);
    }
  };

  // 이미지 업로드 후 에디터에 삽입
  const handleImageUploaded = (imageUrl: string) => {
    const imageMarkdown = `![](${imageUrl})`;
    setContent((prev) => prev + '\n\n' + imageMarkdown);
    setShowImageUploader(false);
  };

  // 모바일 대응: 화면 크기에 따라 viewMode 자동 조정
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768) {
        setViewMode('edit');
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div className="editor-container">
      <div className="editor-header">
        <h2>{post ? '포스트 수정' : `새 ${postType === 'post' ? '포스트' : '노트'}`}</h2>
        <div className="editor-actions">
          {saveStatus && <span className="save-status">{saveStatus}</span>}
          <DraftManager onLoadDraft={handleLoadDraft} currentDraftId={draftId} />
          <button onClick={onCancel} className="btn-cancel">
            취소
          </button>
          <button onClick={handleSave} className="btn-save" disabled={saving}>
            {saving ? '저장 중...' : '저장 및 퍼블리시'}
          </button>
        </div>
      </div>

      <div className="editor-content">
        {/* Frontmatter 폼 */}
        <div className="editor-section">
          <h3>메타데이터</h3>
          <FrontmatterForm frontmatter={frontmatter} onChange={setFrontmatter} />
        </div>

        {/* 이미지 업로더 */}
        <div className="editor-section">
          <div className="section-header">
            <h3>이미지 업로드</h3>
            <button
              onClick={() => setShowImageUploader(!showImageUploader)}
              className="btn-toggle"
            >
              {showImageUploader ? '숨기기' : '보이기'}
            </button>
          </div>
          {showImageUploader && (
            <ImageUploader
              postSlug={
                frontmatter.slug.replace(/^\//, '') || `draft-${draftId}`
              }
              postType={postType}
              onImageUploaded={handleImageUploaded}
            />
          )}
        </div>

        {/* 마크다운 에디터 */}
        <div className="editor-section editor-main">
          <div className="section-header">
            <h3>본문</h3>
            {/* 모바일: 탭 전환 */}
            <div className="view-mode-toggle">
              <button
                onClick={() => setViewMode('edit')}
                className={viewMode === 'edit' ? 'active' : ''}
              >
                편집
              </button>
              <button
                onClick={() => setViewMode('preview')}
                className={viewMode === 'preview' ? 'active' : ''}
              >
                미리보기
              </button>
              <button
                onClick={() => setViewMode('split')}
                className={viewMode === 'split' ? 'active' : ''}
              >
                분할
              </button>
            </div>
          </div>

          <MDEditor
            value={content}
            onChange={(val) => setContent(val || '')}
            preview={viewMode === 'edit' ? 'edit' : viewMode === 'preview' ? 'preview' : 'live'}
            height={500}
            visibleDragbar={false}
          />
        </div>
      </div>
    </div>
  );
};

export default Editor;
