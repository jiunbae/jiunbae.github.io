import { useState, useEffect, useCallback, useRef } from "react";
import FrontmatterForm from "../components/FrontmatterForm";
import ReviewFields from "../components/ReviewFields";
import MarkdownEditor from "../components/MarkdownEditor";
import ImageUploader from "../components/ImageUploader";
import DraftManager from "../components/DraftManager";
import ConfirmDialog from "../components/ConfirmDialog";
import { useGitHubAPI } from "../hooks/useGitHub";
import { useAutoSave } from "../hooks/useAutoSave";
import { parseFrontmatter, serializeFrontmatter } from "../lib/frontmatter";
import {
  generateContentPath,
  parseContentPath,
  CONTENT_DIRS,
  type ContentType,
} from "../lib/content-paths";
import { isContentPath } from "../lib/github-api";
import { sanitizeSlug } from "@/utils/slug";
import type { Draft } from "@/utils/storage";

interface EditorViewProps {
  path?: string;
  contentType: ContentType;
  onBack: () => void;
}

interface Frontmatter {
  title: string;
  description: string;
  date: string;
  slug: string;
  tags: string[];
  published: boolean;
  heroImage?: string;
  heroImageAlt?: string;
  rating?: number;
}

const createDefaultFrontmatter = (): Frontmatter => ({
  title: "",
  description: "",
  date: new Date().toISOString().slice(0, 10),
  slug: "",
  tags: [],
  published: true,
  heroImage: "",
  heroImageAlt: "",
});

const typeToSingular = (t: ContentType) =>
  t === "posts" ? "post" : t === "notes" ? "note" : "review";

export default function EditorView({ path, contentType, onBack }: EditorViewProps) {
  const { fetchContent, saveContent, deleteContent } = useGitHubAPI();
  const draftIdRef = useRef(path ?? `new-${Date.now()}`);

  const [frontmatter, setFrontmatter] = useState<Frontmatter>(createDefaultFrontmatter);
  const [body, setBody] = useState("");
  const [isNew, setIsNew] = useState(!path);
  const [sha, setSha] = useState<string | undefined>(undefined);
  const [saving, setSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  const { lastSaved } = useAutoSave(
    {
      title: frontmatter.title,
      content: body,
      frontmatter,
      type: typeToSingular(contentType),
    },
    draftIdRef.current,
    true,
  );

  // Load existing content
  useEffect(() => {
    if (!path) return;
    let cancelled = false;

    const load = async () => {
      try {
        const file = await fetchContent(path);
        if (cancelled) return;

        const parsed = parseFrontmatter(file.content);
        const fm = parsed.frontmatter;
        const pathInfo = parseContentPath(path);

        setFrontmatter({
          title: fm.title ?? "",
          description: fm.description ?? "",
          date: fm.date
            ? String(fm.date).slice(0, 10)
            : pathInfo?.date ?? "",
          slug: fm.slug ?? pathInfo?.slug ?? "",
          tags: Array.isArray(fm.tags) ? fm.tags : [],
          published: fm.published !== false,
          heroImage: fm.heroImage ?? "",
          heroImageAlt: fm.heroImageAlt ?? "",
          ...(contentType === "reviews" && fm.rating != null
            ? { rating: fm.rating }
            : {}),
        });
        setBody(parsed.body);
        setSha(file.sha);
        setIsNew(false);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to load content",
        );
      }
    };

    load();
    return () => { cancelled = true; };
  }, [path, fetchContent, contentType]);

  const handleSave = useCallback(async () => {
    if (!frontmatter.title.trim()) {
      setError("Title is required");
      return;
    }

    setSaving(true);
    setError(null);
    setSaveStatus(null);

    try {
      const slug = sanitizeSlug(frontmatter.slug || frontmatter.title);
      const date = frontmatter.date || new Date().toISOString().slice(0, 10);

      const targetPath =
        path && !isNew ? path : generateContentPath(contentType, slug, date);

      // H1: Validate path stays within content directories
      if (!isContentPath(targetPath) || targetPath.includes("..")) {
        setError("Invalid content path. Path must be within content directories.");
        setSaving(false);
        return;
      }

      const { rating, heroImage, heroImageAlt, ...baseFm } = { ...frontmatter, slug };
      const fmToSerialize: Record<string, unknown> = { ...baseFm };
      if (contentType === "reviews" && rating != null) {
        fmToSerialize.rating = rating;
      }
      if (heroImage) {
        fmToSerialize.heroImage = heroImage;
        if (heroImageAlt) fmToSerialize.heroImageAlt = heroImageAlt;
      }

      const content = serializeFrontmatter(fmToSerialize, body);
      const action = isNew ? "Create" : "Update";
      const message = `${action} ${contentType}: ${frontmatter.title}`;

      const result = await saveContent(targetPath, content, message, sha);
      setSha(result.sha);
      setIsNew(false);
      setSaveStatus("Saved successfully");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  }, [frontmatter, body, path, isNew, sha, contentType, saveContent]);

  const handleDelete = useCallback(async () => {
    if (!path || !sha) return;
    setShowDeleteConfirm(false);
    setSaving(true);
    setError(null);

    try {
      await deleteContent(path, sha, `Delete ${contentType}: ${frontmatter.title}`);
      onBack();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete");
      setSaving(false);
    }
  }, [path, sha, contentType, frontmatter.title, deleteContent, onBack]);

  const handleImageInsert = useCallback((markdownImage: string) => {
    setBody((prev) => prev + "\n" + markdownImage + "\n");
  }, []);

  const handleLoadDraft = useCallback((draft: Draft) => {
    const defaults = createDefaultFrontmatter();
    const fm = draft.frontmatter;
    setFrontmatter({
      ...defaults,
      title: fm.title ?? defaults.title,
      description: fm.description ?? defaults.description,
      date: fm.date ?? defaults.date,
      slug: fm.slug ?? defaults.slug,
      tags: fm.tags ?? defaults.tags,
      published: fm.heroImage !== undefined ? true : defaults.published,
      heroImage: fm.heroImage,
      heroImageAlt: fm.heroImageAlt,
    });
    setBody(draft.content);
  }, []);

  return (
    <div className="editor-container">
      <div className="editor-header">
        <h2>{isNew ? "New" : "Edit"} {contentType.slice(0, -1)}</h2>
        <div className="editor-actions">
          <button className="btn-cancel" onClick={onBack}>
            Back
          </button>
          <DraftManager onLoad={handleLoadDraft} currentDraftId={draftIdRef.current} />
          {!isNew && path && sha && (
            <button
              className="btn-delete"
              onClick={() => setShowDeleteConfirm(true)}
            >
              Delete
            </button>
          )}
          {saveStatus && <span className="save-status">{saveStatus}</span>}
          {lastSaved && (
            <span className="save-status save-status-dim">
              Draft: {lastSaved.toLocaleTimeString()}
            </span>
          )}
          <button
            className="btn-save"
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? "Saving..." : isNew ? "Create" : "Update"}
          </button>
        </div>
      </div>

      <div className="auth-warning editor-warning">
        main branch commit
      </div>

      {error && <div className="auth-error editor-error"><p>{error}</p></div>}

      <div className="editor-content">
        <div className="editor-section">
          <h3>Metadata</h3>
          <FrontmatterForm
            frontmatter={frontmatter}
            onChange={setFrontmatter}
            contentType={contentType}
          />
          {contentType === "reviews" && (
            <ReviewFields
              rating={frontmatter.rating ?? 0}
              onChange={(rating) =>
                setFrontmatter((prev) => ({ ...prev, rating }))
              }
            />
          )}
        </div>

        <div className="editor-section">
          <h3>Content</h3>
          <MarkdownEditor value={body} onChange={setBody} />
        </div>

        <div className="editor-section">
          <h3>Images</h3>
          <ImageUploader
            onUpload={handleImageInsert}
            contentType={contentType}
            slug={frontmatter.slug || sanitizeSlug(frontmatter.title || "untitled")}
            date={frontmatter.date || new Date().toISOString().slice(0, 10)}
          />
        </div>
      </div>

      <ConfirmDialog
        isOpen={showDeleteConfirm}
        title="Delete content"
        message={`Are you sure you want to delete "${frontmatter.title}"? This cannot be undone.`}
        confirmLabel="Delete"
        onConfirm={handleDelete}
        onCancel={() => setShowDeleteConfirm(false)}
        danger
      />
    </div>
  );
}
