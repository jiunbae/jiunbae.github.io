/**
 * Frontmatter 메타데이터 입력 폼
 * 제목, 설명, 날짜, 슬러그, 태그 등
 */

import React from 'react';
import { Post } from '@/utils/github';

interface FrontmatterFormProps {
  frontmatter: Omit<Post, 'content' | 'path' | 'sha'>;
  onChange: (frontmatter: Omit<Post, 'content' | 'path' | 'sha'>) => void;
  postType: 'post' | 'note';
}

const FrontmatterForm: React.FC<FrontmatterFormProps> = ({ frontmatter, onChange, postType }) => {
  const handleChange = (field: string, value: any) => {
    onChange({
      ...frontmatter,
      [field]: value,
    });
  };

  const handleTagsChange = (tagsString: string) => {
    const tags = tagsString
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean);
    handleChange('tags', tags);
  };

  const generateSlug = () => {
    const slug = frontmatter.title
      .toLowerCase()
      .replace(/[^a-z0-9가-힣\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .trim();

    // notes의 경우 /notes/ prefix 추가
    const finalSlug = postType === 'note' ? `/notes/${slug}` : `/${slug}`;
    handleChange('slug', finalSlug);
  };

  return (
    <div className="frontmatter-form">
      <div className="form-group">
        <label htmlFor="title">제목 *</label>
        <input
          id="title"
          type="text"
          value={frontmatter.title}
          onChange={(e) => handleChange('title', e.target.value)}
          placeholder="포스트 제목"
          required
        />
      </div>

      <div className="form-group">
        <label htmlFor="description">설명</label>
        <textarea
          id="description"
          value={frontmatter.description}
          onChange={(e) => handleChange('description', e.target.value)}
          placeholder="포스트에 대한 간단한 설명"
          rows={2}
        />
      </div>

      <div className="form-row">
        <div className="form-group">
          <label htmlFor="date">날짜 *</label>
          <input
            id="date"
            type="date"
            value={frontmatter.date}
            onChange={(e) => handleChange('date', e.target.value)}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="slug">슬러그 *</label>
          <div className="slug-input-wrapper">
            <input
              id="slug"
              type="text"
              value={frontmatter.slug}
              onChange={(e) => handleChange('slug', e.target.value)}
              placeholder="/my-post"
              required
            />
            <button type="button" onClick={generateSlug} className="btn-generate-slug">
              자동 생성
            </button>
          </div>
        </div>
      </div>

      <div className="form-group">
        <label htmlFor="tags">태그</label>
        <input
          id="tags"
          type="text"
          value={frontmatter.tags.join(', ')}
          onChange={(e) => handleTagsChange(e.target.value)}
          placeholder="react, typescript, web (쉼표로 구분)"
        />
        <small>쉼표(,)로 구분하여 입력</small>
      </div>

      <div className="form-group">
        <label htmlFor="heroImage">히어로 이미지 URL</label>
        <input
          id="heroImage"
          type="text"
          value={frontmatter.heroImage || ''}
          onChange={(e) => handleChange('heroImage', e.target.value)}
          placeholder="/images/hero.jpg"
        />
      </div>

      <div className="form-group">
        <label htmlFor="heroImageAlt">히어로 이미지 설명</label>
        <input
          id="heroImageAlt"
          type="text"
          value={frontmatter.heroImageAlt || ''}
          onChange={(e) => handleChange('heroImageAlt', e.target.value)}
          placeholder="이미지 설명 (접근성)"
        />
      </div>

      <div className="form-group">
        <label htmlFor="published">
          <input
            id="published"
            type="checkbox"
            checked={frontmatter.published !== false}
            onChange={(e) => handleChange('published', e.target.checked)}
            style={{ marginRight: '8px', width: 'auto' }}
          />
          블로그에 공개
        </label>
        <small>체크 해제하면 블로그에 표시되지 않습니다</small>
      </div>
    </div>
  );
};

export default FrontmatterForm;
