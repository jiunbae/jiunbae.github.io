export interface ParsedFrontmatter {
  frontmatter: Record<string, any>;
  body: string;
}

export function parseFrontmatter(markdown: string): ParsedFrontmatter {
  const trimmed = markdown.trimStart();

  if (!trimmed.startsWith("---")) {
    return { frontmatter: {}, body: markdown };
  }

  const endIndex = trimmed.indexOf("\n---", 3);
  if (endIndex === -1) {
    return { frontmatter: {}, body: markdown };
  }

  const yamlBlock = trimmed.slice(4, endIndex).trim();
  const body = trimmed.slice(endIndex + 4).replace(/^\r?\n/, "");
  const frontmatter = parseYaml(yamlBlock);

  return { frontmatter, body };
}

export function serializeFrontmatter(
  frontmatter: Record<string, any>,
  body: string,
): string {
  const lines: string[] = [];

  for (const [key, value] of Object.entries(frontmatter)) {
    if (value === undefined || value === null || value === "") {
      continue;
    }

    lines.push(serializeValue(key, value));
  }

  const fm = lines.length > 0 ? lines.join("\n") + "\n" : "";
  return `---\n${fm}---\n${body}`;
}

function serializeValue(key: string, value: any): string {
  if (typeof value === "boolean") {
    return `${key}: ${value}`;
  }

  if (Array.isArray(value)) {
    const items = value.map((item) => serializeScalar(item)).join(", ");
    return `${key}: [${items}]`;
  }

  if (typeof value === "number") {
    return `${key}: ${value}`;
  }

  const str = String(value);
  if (needsQuoting(str)) {
    return `${key}: "${escapeYamlString(str)}"`;
  }

  return `${key}: ${str}`;
}

function serializeScalar(value: any): string {
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  const str = String(value);
  if (needsQuoting(str) || str.includes(",")) {
    return `"${escapeYamlString(str)}"`;
  }
  return str;
}

function needsQuoting(str: string): boolean {
  if (str === "") return true;
  if (str.includes(":") || str.includes("#") || str.includes('"')) return true;
  if (str.startsWith("'") || str.startsWith('"')) return true;
  if (/^(true|false|yes|no|null)$/i.test(str)) return true;
  return false;
}

function escapeYamlString(str: string): string {
  return str.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function parseYaml(yaml: string): Record<string, any> {
  const result: Record<string, any> = {};
  const lines = yaml.split("\n");

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Skip empty lines and comments
    if (!line.trim() || line.trim().startsWith("#")) continue;

    const colonIndex = line.indexOf(":");
    if (colonIndex === -1) continue;

    const key = line.slice(0, colonIndex).trim();
    let rawValue = line.slice(colonIndex + 1).trim();

    // Check for multiline list (next lines start with -)
    if (rawValue === "" && i + 1 < lines.length && lines[i + 1].trim().startsWith("-")) {
      const items: any[] = [];
      while (i + 1 < lines.length && lines[i + 1].trim().startsWith("-")) {
        i++;
        items.push(parseScalar(lines[i].trim().slice(1).trim()));
      }
      result[key] = items;
      continue;
    }

    result[key] = parseValue(rawValue);
  }

  return result;
}

function parseValue(raw: string): any {
  if (raw === "" || raw === "null" || raw === "~") return null;

  // Inline array: [item1, item2]
  if (raw.startsWith("[") && raw.endsWith("]")) {
    const inner = raw.slice(1, -1).trim();
    if (inner === "") return [];
    return splitArrayItems(inner).map((item) => parseScalar(item.trim()));
  }

  return parseScalar(raw);
}

function splitArrayItems(inner: string): string[] {
  const items: string[] = [];
  let current = "";
  let inQuote: string | null = null;

  for (let i = 0; i < inner.length; i++) {
    const ch = inner[i];

    if (inQuote) {
      if (ch === inQuote && inner[i - 1] !== "\\") {
        inQuote = null;
      }
      current += ch;
    } else if (ch === '"' || ch === "'") {
      inQuote = ch;
      current += ch;
    } else if (ch === ",") {
      items.push(current.trim());
      current = "";
    } else {
      current += ch;
    }
  }

  if (current.trim()) {
    items.push(current.trim());
  }

  return items;
}

function parseScalar(raw: string): any {
  if (raw === "true" || raw === "True") return true;
  if (raw === "false" || raw === "False") return false;
  if (raw === "null" || raw === "~") return null;

  // Quoted string
  if (
    (raw.startsWith('"') && raw.endsWith('"')) ||
    (raw.startsWith("'") && raw.endsWith("'"))
  ) {
    return raw.slice(1, -1).replace(/\\"/g, '"').replace(/\\'/g, "'");
  }

  // Number
  if (/^-?\d+$/.test(raw)) return parseInt(raw, 10);
  if (/^-?\d+\.\d+$/.test(raw)) return parseFloat(raw);

  return raw;
}
