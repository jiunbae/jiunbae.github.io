import { useState, useEffect, useCallback, useRef } from "react";
import { useGitHubAPI } from "../hooks/useGitHub";
import { parseContentPath, CONTENT_DIRS, type ContentType } from "../lib/content-paths";
import { parseFrontmatter } from "../lib/frontmatter";
import ContentItem from "../components/ContentItem";

interface ListViewProps {
  onEdit: (path: string) => void;
  onNew: (type: ContentType) => void;
}

interface ContentMeta {
  title: string;
  description?: string;
  date?: string;
  tags?: string[];
  published?: boolean;
}

const TABS: { key: ContentType; label: string }[] = [
  { key: "posts", label: "Posts" },
  { key: "notes", label: "Notes" },
  { key: "reviews", label: "Reviews" },
];

export default function ListView({ onEdit, onNew }: ListViewProps) {
  const { fetchTree, fetchContent, tree, loading, error } = useGitHubAPI();
  const [activeTab, setActiveTab] = useState<ContentType>("posts");
  const [metaMap, setMetaMap] = useState<Map<string, ContentMeta>>(new Map());
  const [metaLoading, setMetaLoading] = useState<Set<string>>(new Set());
  const fetchedRef = useRef(false);

  useEffect(() => {
    if (!fetchedRef.current) {
      fetchedRef.current = true;
      fetchTree();
    }
  }, [fetchTree]);

  const filteredEntries = (tree ?? [])
    .filter((entry) => {
      const dir = CONTENT_DIRS[activeTab];
      return entry.path.startsWith(dir + "/");
    })
    .filter((entry) => parseContentPath(entry.path) !== null)
    .sort((a, b) => {
      const parsedA = parseContentPath(a.path);
      const parsedB = parseContentPath(b.path);
      if (!parsedA || !parsedB) return 0;
      return parsedB.date.localeCompare(parsedA.date);
    });

  // Batch-fetch frontmatter for visible entries (shared cache via useGitHubAPI)
  useEffect(() => {
    const toFetch = filteredEntries.filter(
      (e) => !metaMap.has(e.path) && !metaLoading.has(e.path),
    );
    if (toFetch.length === 0) return;

    const paths = toFetch.map((e) => e.path);
    setMetaLoading((prev) => {
      const next = new Set(prev);
      paths.forEach((p) => next.add(p));
      return next;
    });

    // Fetch in sequence with small concurrency to avoid rate limits
    const fetchBatch = async () => {
      const results = new Map<string, ContentMeta>();

      for (const path of paths) {
        try {
          const file = await fetchContent(path);
          const { frontmatter } = parseFrontmatter(file.content);
          const parsed = parseContentPath(path);
          results.set(path, {
            title: frontmatter.title ?? parsed?.slug ?? "Untitled",
            description: frontmatter.description,
            date: frontmatter.date
              ? String(frontmatter.date).slice(0, 10)
              : parsed?.date,
            tags: Array.isArray(frontmatter.tags) ? frontmatter.tags : [],
            published: frontmatter.published !== false,
          });
        } catch {
          const parsed = parseContentPath(path);
          results.set(path, {
            title: parsed?.slug ?? "Unknown",
            date: parsed?.date,
          });
        }
      }

      setMetaMap((prev) => {
        const next = new Map(prev);
        results.forEach((v, k) => next.set(k, v));
        return next;
      });
      setMetaLoading((prev) => {
        const next = new Set(prev);
        paths.forEach((p) => next.delete(p));
        return next;
      });
    };

    fetchBatch();
  }, [filteredEntries.map((e) => e.path).join(",")]);

  const handleRefresh = useCallback(() => {
    setMetaMap(new Map());
    setMetaLoading(new Set());
    fetchedRef.current = false;
    fetchTree();
  }, [fetchTree]);

  return (
    <div className="post-list">
      <div className="post-list-header">
        <div className="post-list-tabs">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              className={`tab${activeTab === tab.key ? " active" : ""}`}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <div className="post-list-actions">
          <button className="btn-new" onClick={() => onNew(activeTab)}>
            New
          </button>
          <button className="btn-refresh" onClick={handleRefresh} disabled={loading}>
            Refresh
          </button>
        </div>
      </div>

      <div className="post-list-content">
        {loading && !tree && (
          <div className="post-list-loading">Loading...</div>
        )}

        {!loading && error && (
          <div className="post-list-empty">{error}</div>
        )}

        {!loading && !error && filteredEntries.length === 0 && (
          <div className="post-list-empty">No content found.</div>
        )}

        {filteredEntries.length > 0 && (
          <div className="post-list-items">
            {filteredEntries.map((entry) => (
              <ContentItem
                key={entry.path}
                path={entry.path}
                onClick={() => onEdit(entry.path)}
                meta={metaMap.get(entry.path)}
                loading={metaLoading.has(entry.path)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
