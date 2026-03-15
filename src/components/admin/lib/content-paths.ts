export type ContentType = "posts" | "notes" | "reviews";

export const CONTENT_DIRS: Record<ContentType, string> = {
  posts: "src/content/posts",
  notes: "src/content/notes",
  reviews: "src/content/reviews",
};

export function getContentDir(type: ContentType): string {
  return CONTENT_DIRS[type];
}

export function generateContentPath(
  type: ContentType,
  slug: string,
  date: string,
): string {
  const dateStr = formatDate(date);
  const dir = CONTENT_DIRS[type];

  if (type === "notes") {
    return `${dir}/${dateStr}-${slug}.md`;
  }

  // posts and reviews use directory-based structure
  return `${dir}/${dateStr}-${slug}/index.md`;
}

export function generateImagePath(
  type: ContentType,
  slug: string,
  date: string,
  filename: string,
): string {
  const dateStr = formatDate(date);
  const dir = CONTENT_DIRS[type];

  if (type === "notes") {
    // Notes are flat files, prefix image with date-slug to avoid collisions
    return `${dir}/${dateStr}-${slug}-${filename}`;
  }

  return `${dir}/${dateStr}-${slug}/${filename}`;
}

export interface ParsedContentPath {
  type: ContentType;
  slug: string;
  date: string;
}

export function parseContentPath(path: string): ParsedContentPath | null {
  for (const [type, dir] of Object.entries(CONTENT_DIRS) as [ContentType, string][]) {
    if (!path.startsWith(dir + "/")) continue;

    const relative = path.slice(dir.length + 1);

    if (type === "notes") {
      // Pattern: {YYYY-MM-DD}-{slug}.md
      const match = relative.match(/^(\d{4}-\d{2}-\d{2})-(.+)\.md$/);
      if (match) {
        return { type, date: match[1], slug: match[2] };
      }
    } else {
      // Pattern: {YYYY-MM-DD}-{slug}/index.md
      const match = relative.match(/^(\d{4}-\d{2}-\d{2})-(.+)\/index\.md$/);
      if (match) {
        return { type, date: match[1], slug: match[2] };
      }
    }
  }

  return null;
}

function formatDate(date: string): string {
  // If already in YYYY-MM-DD format, return as-is
  if (/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    return date;
  }

  // Parse and format
  const d = new Date(date);
  if (isNaN(d.getTime())) {
    // Fallback to today if date is invalid
    const now = new Date();
    return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
  }
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}
