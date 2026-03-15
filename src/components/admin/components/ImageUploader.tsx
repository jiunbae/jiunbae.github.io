import { useState, useCallback, useRef, type DragEvent, type ChangeEvent } from "react";
import { generateImagePath, type ContentType } from "../lib/content-paths";
import { useGitHubAPI } from "../hooks/useGitHub";

interface ImageUploaderProps {
  onUpload: (markdownImage: string) => void;
  contentType: "posts" | "notes" | "reviews";
  slug: string;
  date: string;
}

export default function ImageUploader({
  onUpload,
  contentType,
  slug,
  date,
}: ImageUploaderProps) {
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { uploadImage } = useGitHubAPI();

  const processFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith("image/")) return;

      setUploading(true);
      try {
        const base64 = await fileToBase64(file);
        const path = generateImagePath(
          contentType as ContentType,
          slug,
          date,
          file.name,
        );
        await uploadImage(path, base64, `Upload image: ${file.name}`);

        const relativePath = `./${file.name}`;
        const altText = file.name.replace(/\.[^.]+$/, "");
        onUpload(`![${altText}](${relativePath})`);
      } catch (err) {
        console.error("Image upload failed:", err);
      } finally {
        setUploading(false);
      }
    },
    [contentType, slug, date, uploadImage, onUpload],
  );

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        processFile(files[0]);
      }
    },
    [processFile],
  );

  const handleFileSelect = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        processFile(files[0]);
      }
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    },
    [processFile],
  );

  const handleClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  return (
    <div className="image-uploader">
      <div
        className={`dropzone${dragActive ? " active" : ""}${uploading ? " uploading" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        {uploading ? (
          <div className="dropzone-uploading">Uploading...</div>
        ) : (
          <div className="dropzone-content">
            Drop an image here or click to select
          </div>
        )}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={handleFileSelect}
      />
    </div>
  );
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Strip the data URL prefix (e.g., "data:image/png;base64,")
      const base64 = result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
