import { useState, useEffect, useCallback } from "react";
import { useGitHubAPI } from "../hooks/useGitHub";
import { parseContentPath, CONTENT_DIRS, type ContentType } from "../lib/content-paths";
import ContentItem from "../components/ContentItem";

interface ListViewProps {
  onEdit: (path: string) => void;
  onNew: (type: ContentType) => void;
}

const TABS: { key: ContentType; label: string }[] = [
  { key: "posts", label: "Posts" },
  { key: "notes", label: "Notes" },
  { key: "reviews", label: "Reviews" },
];

export default function ListView({ onEdit, onNew }: ListViewProps) {
  const { fetchTree, tree, loading, error } = useGitHubAPI();
  const [activeTab, setActiveTab] = useState<ContentType>("posts");

  useEffect(() => {
    fetchTree();
  }, []);

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
          <button className="btn-refresh" onClick={fetchTree} disabled={loading}>
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
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
