import { useState, useEffect } from "react";
import { parseContentPath } from "../lib/content-paths";
import { parseFrontmatter } from "../lib/frontmatter";
import { useGitHubAPI } from "../hooks/useGitHub";

interface ContentItemProps {
  path: string;
  onClick: () => void;
}

interface ContentMeta {
  title: string;
  description?: string;
  date?: string;
  tags?: string[];
  published?: boolean;
}

export default function ContentItem({ path, onClick }: ContentItemProps) {
  const [meta, setMeta] = useState<ContentMeta | null>(null);
  const [loading, setLoading] = useState(true);
  const { fetchContent } = useGitHubAPI();

  const parsed = parseContentPath(path);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      try {
        const { content } = await fetchContent(path);
        const { frontmatter } = parseFrontmatter(content);

        if (!cancelled) {
          setMeta({
            title: frontmatter.title ?? parsed?.slug ?? "Untitled",
            description: frontmatter.description,
            date: frontmatter.date ?? parsed?.date,
            tags: Array.isArray(frontmatter.tags) ? frontmatter.tags : [],
            published: frontmatter.published !== false,
          });
        }
      } catch {
        if (!cancelled) {
          setMeta({
            title: parsed?.slug ?? path.split("/").pop() ?? "Unknown",
            date: parsed?.date,
          });
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [path, fetchContent]);

  if (loading) {
    return (
      <div className="post-list-item" onClick={onClick}>
        <div className="post-item-header">
          <span className="post-item-title" style={{ opacity: 0.5 }}>
            Loading...
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="post-list-item" onClick={onClick}>
      <div className="post-item-header">
        <span className="post-item-title">{meta?.title ?? "Untitled"}</span>
        {meta?.published === false && (
          <span className="unpublished-badge">Unpublished</span>
        )}
      </div>
      {meta?.description && (
        <p className="post-item-description">{meta.description}</p>
      )}
      <div className="post-item-meta">
        {meta?.date && <span>{meta.date}</span>}
        {meta?.tags && meta.tags.length > 0 && (
          <div className="post-item-tags">
            {meta.tags.map((tag) => (
              <span key={tag} className="tag">
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
