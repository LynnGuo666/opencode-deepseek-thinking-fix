/**
 * In-memory cache mapping an "assistant turn fingerprint" -> the original
 * thinking content blocks (with signatures) returned by the upstream.
 *
 * Why fingerprint instead of sessionId:
 * OpenCode does not necessarily expose a stable sessionId to the provider
 * fetch layer, and multiple concurrent sessions may share one fetch.
 * We fingerprint by hashing the prefix of messages that precede the assistant
 * turn we want to preserve.
 */

export interface ThinkingBlock {
  type: "thinking" | "redacted_thinking";
  thinking?: string;
  data?: string;
  signature?: string;
  [k: string]: unknown;
}

export interface CacheEntry {
  blocks: ThinkingBlock[];
  createdAt: number;
}

const DEFAULT_TTL_MS = 30 * 60 * 1000;
const MAX_ENTRIES = 500;

export class ThinkingCache {
  private store = new Map<string, CacheEntry>();
  constructor(private ttlMs: number = DEFAULT_TTL_MS) {}

  set(key: string, blocks: ThinkingBlock[]): void {
    if (!blocks.length) return;
    if (this.store.size >= MAX_ENTRIES) {
      // evict oldest
      const oldest = [...this.store.entries()].sort(
        (a, b) => a[1].createdAt - b[1].createdAt,
      )[0];
      if (oldest) this.store.delete(oldest[0]);
    }
    this.store.set(key, { blocks, createdAt: Date.now() });
  }

  get(key: string): ThinkingBlock[] | undefined {
    const entry = this.store.get(key);
    if (!entry) return;
    if (Date.now() - entry.createdAt > this.ttlMs) {
      this.store.delete(key);
      return;
    }
    return entry.blocks;
  }

  clear(): void {
    this.store.clear();
  }
}

/**
 * Stable FNV-1a 64-bit-ish hash of a JSON-serializable value.
 * Good enough for fingerprinting message prefixes.
 */
export function fingerprint(value: unknown): string {
  const s = typeof value === "string" ? value : stableStringify(value);
  let h1 = 0xcbf29ce4;
  let h2 = 0x84222325;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    h1 ^= c;
    h1 = (h1 + ((h1 << 1) + (h1 << 4) + (h1 << 7) + (h1 << 8) + (h1 << 24))) >>> 0;
    h2 ^= c;
    h2 = (h2 + ((h2 << 1) + (h2 << 4) + (h2 << 7) + (h2 << 8) + (h2 << 24))) >>> 0;
  }
  return h1.toString(16).padStart(8, "0") + h2.toString(16).padStart(8, "0");
}

function stableStringify(v: unknown): string {
  if (v === null || typeof v !== "object") return JSON.stringify(v);
  if (Array.isArray(v)) return "[" + v.map(stableStringify).join(",") + "]";
  const keys = Object.keys(v as Record<string, unknown>).sort();
  return (
    "{" +
    keys
      .map(
        (k) =>
          JSON.stringify(k) +
          ":" +
          stableStringify((v as Record<string, unknown>)[k]),
      )
      .join(",") +
    "}"
  );
}
