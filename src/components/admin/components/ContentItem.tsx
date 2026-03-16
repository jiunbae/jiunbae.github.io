import { parseContentPath } from "../lib/content-paths";

interface ContentItemProps {
  path: string;
  onClick: () => void;
  meta?: {
    title: string;
    description?: string;
    date?: string;
    tags?: string[];
    published?: boolean;
  } | null;
  loading?: boolean;
}

export default function ContentItem({ path, onClick, meta, loading }: ContentItemProps) {
  const parsed = parseContentPath(path);

  if (loading) {
    return (
      <div className="post-list-item" onClick={onClick}>
        <div className="post-item-header">
          <span className="post-item-title" style={{ opacity: 0.5 }}>
            {parsed?.slug ?? "Loading..."}
          </span>
        </div>
        <div className="post-item-meta">
          {parsed?.date && <span>{parsed.date}</span>}
        </div>
      </div>
    );
  }

  const title = meta?.title ?? parsed?.slug ?? "Untitled";
  const date = meta?.date ?? parsed?.date;

  return (
    <div className="post-list-item" onClick={onClick}>
      <div className="post-item-header">
        <span className="post-item-title">{title}</span>
        {meta?.published === false && (
          <span className="unpublished-badge">Unpublished</span>
        )}
      </div>
      {meta?.description && (
        <p className="post-item-description">{meta.description}</p>
      )}
      <div className="post-item-meta">
        {date && <span>{date}</span>}
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
