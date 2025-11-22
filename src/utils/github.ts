/**
 * GitHub API 헬퍼 함수
 * Octokit을 사용하여 레포지토리 파일 읽기/쓰기
 */

import { Octokit } from '@octokit/rest';

const REPO_OWNER = 'jiunbae';
const REPO_NAME = 'jiunbae.github.io';
const BRANCH = 'main';

export interface GitHubFile {
  path: string;
  content: string;
  sha?: string;
}

export interface Post {
  title: string;
  description: string;
  date: string;
  slug: string;
  tags: string[];
  heroImage?: string;
  heroImageAlt?: string;
  content: string;
  path: string;
  sha?: string;
}

/**
 * Octokit 인스턴스 생성
 */
export const createOctokit = (token: string): Octokit => {
  return new Octokit({ auth: token });
};

/**
 * GitHub 토큰 유효성 검증
 */
export const validateToken = async (token: string): Promise<boolean> => {
  try {
    const octokit = createOctokit(token);
    await octokit.users.getAuthenticated();
    return true;
  } catch (error) {
    console.error('Token validation failed:', error);
    return false;
  }
};

/**
 * 파일 내용 조회
 */
export const getFileContent = async (
  octokit: Octokit,
  path: string
): Promise<GitHubFile | null> => {
  try {
    const { data } = await octokit.repos.getContent({
      owner: REPO_OWNER,
      repo: REPO_NAME,
      path,
      ref: BRANCH,
    });

    if ('content' in data && data.type === 'file') {
      // Browser-compatible base64 decoding
      const base64Content = data.content.replace(/\n/g, '');
      const decodedContent = atob(base64Content);
      // Convert to UTF-8
      const content = decodeURIComponent(escape(decodedContent));
      return {
        path,
        content,
        sha: data.sha,
      };
    }

    return null;
  } catch (error) {
    console.error(`Failed to get file content: ${path}`, error);
    return null;
  }
};

/**
 * 디렉토리 내용 조회
 */
export const getDirectoryContents = async (
  octokit: Octokit,
  path: string
): Promise<Array<{ name: string; path: string; type: string }>> => {
  try {
    const { data } = await octokit.repos.getContent({
      owner: REPO_OWNER,
      repo: REPO_NAME,
      path,
      ref: BRANCH,
    });

    if (Array.isArray(data)) {
      return data.map((item) => ({
        name: item.name,
        path: item.path,
        type: item.type,
      }));
    }

    return [];
  } catch (error) {
    console.error(`Failed to get directory contents: ${path}`, error);
    return [];
  }
};

/**
 * 포스트 목록 조회
 */
export const getPosts = async (
  octokit: Octokit,
  type: 'posts' | 'notes' = 'posts'
): Promise<Post[]> => {
  try {
    const basePath = `contents/${type}`;
    const items = await getDirectoryContents(octokit, basePath);

    const posts: Post[] = [];

    for (const item of items) {
      if (type === 'posts' && item.type === 'dir') {
        // posts는 디렉토리 형식: contents/posts/YYYY-MM-DD-slug/index.md
        const indexFile = await getFileContent(octokit, `${item.path}/index.md`);
        if (indexFile) {
          const parsed = parseMarkdownFile(indexFile.content);
          posts.push({
            ...parsed.frontmatter,
            content: parsed.content,
            path: `${item.path}/index.md`,
            sha: indexFile.sha,
          });
        }
      } else if (type === 'notes' && item.type === 'file' && item.name.endsWith('.md')) {
        // notes는 파일 형식: contents/notes/YYYY-MM-DD-slug.md
        const file = await getFileContent(octokit, item.path);
        if (file) {
          const parsed = parseMarkdownFile(file.content);
          posts.push({
            ...parsed.frontmatter,
            content: parsed.content,
            path: item.path,
            sha: file.sha,
          });
        }
      }
    }

    return posts.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  } catch (error) {
    console.error('Failed to get posts:', error);
    return [];
  }
};

/**
 * 마크다운 파일 파싱 (frontmatter + content)
 */
export const parseMarkdownFile = (
  content: string
): {
  frontmatter: Omit<Post, 'content' | 'path' | 'sha'>;
  content: string;
} => {
  const frontmatterRegex = /^---\n([\s\S]*?)\n---\n([\s\S]*)$/;
  const match = content.match(frontmatterRegex);

  if (!match) {
    return {
      frontmatter: {
        title: '',
        description: '',
        date: new Date().toISOString().split('T')[0],
        slug: '',
        tags: [],
      },
      content: content,
    };
  }

  const frontmatterText = match[1];
  const bodyContent = match[2];

  const frontmatter: Omit<Post, 'content' | 'path' | 'sha'> = {
    title: '',
    description: '',
    date: new Date().toISOString().split('T')[0],
    slug: '',
    tags: [],
  };

  frontmatterText.split('\n').forEach((line) => {
    const [key, ...valueParts] = line.split(':');
    if (key && valueParts.length > 0) {
      const value = valueParts.join(':').trim();
      const trimmedKey = key.trim() as keyof typeof frontmatter;

      if (trimmedKey === 'tags') {
        // Parse array: [tag1, tag2]
        const arrayMatch = value.match(/\[(.*?)\]/);
        if (arrayMatch) {
          frontmatter.tags = arrayMatch[1]
            .split(',')
            .map((t) => t.trim().replace(/['"]/g, ''))
            .filter(Boolean);
        }
      } else if (trimmedKey === 'heroImage' || trimmedKey === 'heroImageAlt') {
        // Handle optional fields - set undefined if empty
        if (value && value !== '') {
          frontmatter[trimmedKey] = value;
        } else {
          frontmatter[trimmedKey] = undefined;
        }
      } else if (trimmedKey in frontmatter) {
        // Only set known properties
        (frontmatter as Record<string, string | string[] | undefined>)[trimmedKey] = value;
      }
    }
  });

  return {
    frontmatter,
    content: bodyContent.trim(),
  };
};

/**
 * 마크다운 파일 생성 (frontmatter + content)
 */
export const createMarkdownFile = (
  frontmatter: Omit<Post, 'content' | 'path' | 'sha'>,
  content: string
): string => {
  const frontmatterText = `---
title: ${frontmatter.title}
description: ${frontmatter.description}
date: ${frontmatter.date}
slug: ${frontmatter.slug}
tags: [${frontmatter.tags.join(', ')}]${
    frontmatter.heroImage
      ? `
heroImage: ${frontmatter.heroImage}`
      : ''
  }${
    frontmatter.heroImageAlt
      ? `
heroImageAlt: ${frontmatter.heroImageAlt}`
      : ''
  }
---

${content}`;

  return frontmatterText;
};

/**
 * 파일 생성 또는 업데이트
 */
export const createOrUpdateFile = async (
  octokit: Octokit,
  path: string,
  content: string,
  message: string,
  sha?: string
): Promise<boolean> => {
  try {
    // Browser-compatible base64 encoding
    const contentBase64 = btoa(unescape(encodeURIComponent(content)));

    await octokit.repos.createOrUpdateFileContents({
      owner: REPO_OWNER,
      repo: REPO_NAME,
      path,
      message,
      content: contentBase64,
      branch: BRANCH,
      sha,
    });

    return true;
  } catch (error) {
    console.error('Failed to create or update file:', error);
    return false;
  }
};

/**
 * 이미지 업로드
 */
export const uploadImage = async (
  octokit: Octokit,
  path: string,
  imageData: string | ArrayBuffer,
  message: string = 'Upload image'
): Promise<string | null> => {
  try {
    let contentBase64: string;

    if (typeof imageData === 'string') {
      // Base64 string
      contentBase64 = imageData.replace(/^data:image\/\w+;base64,/, '');
    } else {
      // ArrayBuffer - browser-compatible conversion
      const bytes = new Uint8Array(imageData);
      let binary = '';
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      contentBase64 = btoa(binary);
    }

    await octokit.repos.createOrUpdateFileContents({
      owner: REPO_OWNER,
      repo: REPO_NAME,
      path,
      message,
      content: contentBase64,
      branch: BRANCH,
    });

    // Return the relative path for markdown
    return `/${path}`;
  } catch (error) {
    console.error('Failed to upload image:', error);
    return null;
  }
};

/**
 * 파일 삭제
 */
export const deleteFile = async (
  octokit: Octokit,
  path: string,
  sha: string,
  message: string
): Promise<boolean> => {
  try {
    await octokit.repos.deleteFile({
      owner: REPO_OWNER,
      repo: REPO_NAME,
      path,
      message,
      sha,
      branch: BRANCH,
    });

    return true;
  } catch (error) {
    console.error('Failed to delete file:', error);
    return false;
  }
};
