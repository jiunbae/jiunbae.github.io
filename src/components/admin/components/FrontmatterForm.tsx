import { useCallback } from "react";
import { sanitizeSlug } from "@/utils/slug";

interface Frontmatter {
  title: string;
  description: string;
  date: string;
  slug: string;
  tags: string[];
  published: boolean;
  heroImage?: string;
  heroImageAlt?: string;
}

interface FrontmatterFormProps {
  frontmatter: Frontmatter;
  onChange: (fm: Frontmatter) => void;
  contentType: "posts" | "notes" | "reviews";
}

export default function FrontmatterForm({
  frontmatter,
  onChange,
  contentType,
}: FrontmatterFormProps) {
  const update = useCallback(
    (partial: Partial<Frontmatter>) => {
      onChange({ ...frontmatter, ...partial });
    },
    [frontmatter, onChange],
  );

  const handleTagsChange = useCallback(
    (value: string) => {
      const tags = value
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      update({ tags });
    },
    [update],
  );

  const handleGenerateSlug = useCallback(() => {
    update({ slug: sanitizeSlug(frontmatter.title) });
  }, [frontmatter.title, update]);

  return (
    <div className="frontmatter-form">
      <div className="form-group">
        <label htmlFor="fm-title">Title</label>
        <input
          id="fm-title"
          type="text"
          value={frontmatter.title}
          onChange={(e) => update({ title: e.target.value })}
          placeholder="Post title"
        />
      </div>

      <div className="form-group">
        <label htmlFor="fm-description">Description</label>
        <textarea
          id="fm-description"
          value={frontmatter.description}
          onChange={(e) => update({ description: e.target.value })}
          placeholder="Brief description"
          rows={3}
        />
      </div>

      <div className="form-row">
        <div className="form-group">
          <label htmlFor="fm-date">Date</label>
          <input
            id="fm-date"
            type="date"
            value={frontmatter.date}
            onChange={(e) => update({ date: e.target.value })}
          />
        </div>

        <div className="form-group">
          <label htmlFor="fm-slug">Slug</label>
          <div className="slug-input-wrapper">
            <input
              id="fm-slug"
              type="text"
              value={frontmatter.slug}
              onChange={(e) => update({ slug: e.target.value })}
              placeholder="url-friendly-slug"
            />
            <button
              type="button"
              className="btn-generate-slug"
              onClick={handleGenerateSlug}
              title="Generate slug from title"
            >
              Generate
            </button>
          </div>
        </div>
      </div>

      <div className="form-group">
        <label htmlFor="fm-tags">Tags</label>
        <input
          id="fm-tags"
          type="text"
          value={frontmatter.tags.join(", ")}
          onChange={(e) => handleTagsChange(e.target.value)}
          placeholder="tag1, tag2, tag3"
        />
      </div>

      <div className="form-group">
        <label>
          <input
            type="checkbox"
            checked={frontmatter.published}
            onChange={(e) => update({ published: e.target.checked })}
          />
          {" "}Published
        </label>
      </div>

      <div className="form-group">
        <label htmlFor="fm-hero-image">Hero Image URL</label>
        <input
          id="fm-hero-image"
          type="text"
          value={frontmatter.heroImage ?? ""}
          onChange={(e) =>
            update({ heroImage: e.target.value || undefined })
          }
          placeholder="https://example.com/image.jpg (optional)"
        />
      </div>

      {frontmatter.heroImage && (
        <div className="form-group">
          <label htmlFor="fm-hero-image-alt">Hero Image Alt Text</label>
          <input
            id="fm-hero-image-alt"
            type="text"
            value={frontmatter.heroImageAlt ?? ""}
            onChange={(e) =>
              update({ heroImageAlt: e.target.value || undefined })
            }
            placeholder="Image description"
          />
        </div>
      )}
    </div>
  );
}
