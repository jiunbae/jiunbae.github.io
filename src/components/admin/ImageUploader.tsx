/**
 * 이미지 업로더 컴포넌트
 * 드래그앤드롭 이미지 업로드 및 GitHub 저장
 */

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useGitHub } from '@/contexts/GitHubContext';
import { uploadImage } from '@/utils/github';

interface ImageUploaderProps {
  postSlug: string;
  postType: 'post' | 'note';
  onImageUploaded: (imageUrl: string) => void;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({
  postSlug,
  postType,
  onImageUploaded,
}) => {
  const { octokit } = useGitHub();
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState('');

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (!octokit || acceptedFiles.length === 0) return;

      setUploading(true);

      for (const file of acceptedFiles) {
        try {
          setUploadProgress(`업로드 중: ${file.name}`);

          // 이미지를 base64로 변환
          const reader = new FileReader();
          reader.onload = async (e) => {
            if (!e.target?.result) return;

            const base64Data = e.target.result as string;

            // 파일 확장자 추출 및 보안 강화된 파일명 생성
            const timestamp = Date.now();
            const fileExtension = file.name.split('.').pop()?.toLowerCase() || 'jpg';
            // 확장자만 추출하고 타임스탬프 기반으로 파일명 생성 (directory traversal 방지)
            const safeExtension = fileExtension.replace(/[^a-z0-9]/g, '');
            const fileName = `image_${timestamp}.${safeExtension}`;

            // 업로드 경로 생성 (postSlug가 비어있으면 공통 이미지 폴더 사용)
            const slugForPath = postSlug && postSlug !== '' ? postSlug : 'common';
            const uploadPath =
              postType === 'post'
                ? `contents/posts/${slugForPath}/${fileName}`
                : `contents/images/${fileName}`;

            // GitHub에 업로드
            const imageUrl = await uploadImage(
              octokit,
              uploadPath,
              base64Data,
              `Upload image: ${fileName}`
            );

            if (imageUrl) {
              onImageUploaded(imageUrl);
              setUploadProgress(`업로드 완료: ${file.name}`);
              setTimeout(() => setUploadProgress(''), 2000);
            } else {
              setUploadProgress(`업로드 실패: ${file.name}`);
              setTimeout(() => setUploadProgress(''), 3000);
            }
          };

          reader.readAsDataURL(file);
        } catch (error) {
          console.error('Failed to upload image:', error);
          setUploadProgress(`업로드 실패: ${file.name}`);
          setTimeout(() => setUploadProgress(''), 3000);
        }
      }

      setUploading(false);
    },
    [octokit, postSlug, postType, onImageUploaded]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg'],
    },
    multiple: true,
    disabled: uploading,
  });

  return (
    <div className="image-uploader">
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''} ${uploading ? 'uploading' : ''}`}
      >
        <input {...getInputProps()} />
        {uploading ? (
          <div className="dropzone-uploading">
            <p>{uploadProgress}</p>
          </div>
        ) : isDragActive ? (
          <p>이미지를 여기에 놓으세요...</p>
        ) : (
          <div className="dropzone-content">
            <p>이미지를 드래그하거나 클릭하여 업로드</p>
            <small>PNG, JPG, GIF, WebP, SVG 지원</small>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUploader;
