/**
 * Review 메타데이터 입력 폼
 * mediaType, rating, oneLiner, metadata 등
 */

import React from 'react';
import { Review } from '@/utils/github';

interface ReviewFormProps {
  frontmatter: Omit<Review, 'content' | 'path' | 'sha'>;
  onChange: (frontmatter: Omit<Review, 'content' | 'path' | 'sha'>) => void;
}

const ReviewForm: React.FC<ReviewFormProps> = ({ frontmatter, onChange }) => {
  const handleChange = (field: string, value: any) => {
    onChange({
      ...frontmatter,
      [field]: value,
    });
  };

  const handleMetadataChange = (field: string, value: any) => {
    onChange({
      ...frontmatter,
      metadata: {
        ...frontmatter.metadata,
        [field]: value,
      },
    });
  };

  const handleExternalIdsChange = (field: string, value: any) => {
    onChange({
      ...frontmatter,
      externalIds: {
        ...frontmatter.externalIds,
        [field]: value,
      },
    });
  };

  const handleTagsChange = (tagsString: string) => {
    const tags = tagsString
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean);
    handleChange('tags', tags);
  };

  const handleGenreChange = (genreString: string) => {
    const genre = genreString
      .split(',')
      .map((g) => g.trim())
      .filter(Boolean);
    handleMetadataChange('genre', genre);
  };

  const generateSlug = () => {
    const slug = frontmatter.title
      .toLowerCase()
      .replace(/[^a-z0-9가-힣\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .trim();

    handleChange('slug', `/reviews/${slug}`);
  };

  return (
    <div className="review-form">
      <div className="form-group">
        <label htmlFor="title">제목 *</label>
        <input
          id="title"
          type="text"
          value={frontmatter.title}
          onChange={(e) => handleChange('title', e.target.value)}
          placeholder="리뷰 제목"
          required
        />
      </div>

      <div className="form-row">
        <div className="form-group">
          <label htmlFor="mediaType">미디어 타입 *</label>
          <select
            id="mediaType"
            value={frontmatter.mediaType}
            onChange={(e) => handleChange('mediaType', e.target.value)}
            required
          >
            <option value="movie">영화</option>
            <option value="animation">애니메이션</option>
            <option value="tv">TV 시리즈</option>
            <option value="book">책</option>
            <option value="game">게임</option>
          </select>
        </div>

        <div className="form-group">
          <label htmlFor="rating">평점 *</label>
          <input
            id="rating"
            type="number"
            min="0"
            max="5"
            step="0.5"
            value={frontmatter.rating}
            onChange={(e) => handleChange('rating', parseFloat(e.target.value))}
            required
          />
        </div>
      </div>

      <div className="form-group">
        <label htmlFor="oneLiner">한 줄 평 *</label>
        <input
          id="oneLiner"
          type="text"
          value={frontmatter.oneLiner}
          onChange={(e) => handleChange('oneLiner', e.target.value)}
          placeholder="한 줄로 요약"
          required
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
              placeholder="/reviews/my-review"
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
          placeholder="판타지, 모험, 성장 (쉼표로 구분)"
        />
        <small>쉼표(,)로 구분하여 입력</small>
      </div>

      <div className="form-group">
        <label htmlFor="poster">포스터 이미지</label>
        <input
          id="poster"
          type="text"
          value={frontmatter.poster || ''}
          onChange={(e) => handleChange('poster', e.target.value)}
          placeholder="./poster.jpg 또는 /images/poster.jpg"
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

      <hr />

      <h3>메타데이터</h3>

      <div className="form-group">
        <label htmlFor="originalTitle">원제</label>
        <input
          id="originalTitle"
          type="text"
          value={frontmatter.metadata.originalTitle || ''}
          onChange={(e) => handleMetadataChange('originalTitle', e.target.value)}
          placeholder="Original Title"
        />
      </div>

      <div className="form-row">
        <div className="form-group">
          <label htmlFor="year">연도</label>
          <input
            id="year"
            type="number"
            value={frontmatter.metadata.year || ''}
            onChange={(e) => handleMetadataChange('year', parseInt(e.target.value, 10))}
            placeholder="2024"
          />
        </div>

        <div className="form-group">
          <label htmlFor="director">감독/제작자</label>
          <input
            id="director"
            type="text"
            value={frontmatter.metadata.director || frontmatter.metadata.creator || ''}
            onChange={(e) => {
              if (
                frontmatter.mediaType === 'movie' ||
                frontmatter.mediaType === 'animation' ||
                frontmatter.mediaType === 'tv'
              ) {
                handleMetadataChange('director', e.target.value);
              } else {
                handleMetadataChange('creator', e.target.value);
              }
            }}
            placeholder="감독명 또는 제작자명"
          />
        </div>
      </div>

      {frontmatter.mediaType === 'book' && (
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="author">저자</label>
            <input
              id="author"
              type="text"
              value={frontmatter.metadata.author || ''}
              onChange={(e) => handleMetadataChange('author', e.target.value)}
              placeholder="저자명"
            />
          </div>

          <div className="form-group">
            <label htmlFor="pages">페이지</label>
            <input
              id="pages"
              type="number"
              value={frontmatter.metadata.pages || ''}
              onChange={(e) => handleMetadataChange('pages', parseInt(e.target.value, 10))}
              placeholder="300"
            />
          </div>
        </div>
      )}

      {(frontmatter.mediaType === 'movie' ||
        frontmatter.mediaType === 'animation' ||
        frontmatter.mediaType === 'tv') && (
        <div className="form-group">
          <label htmlFor="runtime">러닝타임</label>
          <input
            id="runtime"
            type="text"
            value={frontmatter.metadata.runtime || ''}
            onChange={(e) => handleMetadataChange('runtime', e.target.value)}
            placeholder="120분"
          />
        </div>
      )}

      <div className="form-group">
        <label htmlFor="genre">장르</label>
        <input
          id="genre"
          type="text"
          value={frontmatter.metadata.genre?.join(', ') || ''}
          onChange={(e) => handleGenreChange(e.target.value)}
          placeholder="판타지, 모험, 가족 (쉼표로 구분)"
        />
        <small>쉼표(,)로 구분하여 입력</small>
      </div>

      <hr />

      <h3>외부 ID</h3>

      <div className="form-row">
        <div className="form-group">
          <label htmlFor="tmdbId">TMDB ID</label>
          <input
            id="tmdbId"
            type="text"
            value={frontmatter.externalIds?.tmdbId || ''}
            onChange={(e) => handleExternalIdsChange('tmdbId', e.target.value)}
            placeholder="129"
          />
        </div>

        <div className="form-group">
          <label htmlFor="imdbId">IMDB ID</label>
          <input
            id="imdbId"
            type="text"
            value={frontmatter.externalIds?.imdbId || ''}
            onChange={(e) => handleExternalIdsChange('imdbId', e.target.value)}
            placeholder="tt0245429"
          />
        </div>
      </div>
    </div>
  );
};

export default ReviewForm;
