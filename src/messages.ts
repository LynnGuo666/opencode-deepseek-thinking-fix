import { fingerprint, ThinkingBlock } from "./cache.js";

export interface AnthropicMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string | AnthropicContentBlock[];
  [k: string]: unknown;
}

export type AnthropicContentBlock =
  | { type: "text"; text: string; [k: string]: unknown }
  | { type: "thinking"; thinking?: string; signature?: string; [k: string]: unknown }
  | { type: "redacted_thinking"; data?: string; [k: string]: unknown }
  | { type: "tool_use"; [k: string]: unknown }
  | { type: "tool_result"; [k: string]: unknown }
  | { type: string; [k: string]: unknown };

export interface AnthropicRequestBody {
  model?: string;
  messages?: AnthropicMessage[];
  system?: unknown;
  thinking?: { type?: string; budget_tokens?: number } | null;
  stream?: boolean;
  [k: string]: unknown;
}

export function hasThinkingBlock(msg: AnthropicMessage): boolean {
  if (typeof msg.content === "string") return false;
  return msg.content.some(
    (c) => c.type === "thinking" || c.type === "redacted_thinking",
  );
}

/**
 * Compute a fingerprint for the assistant turn at index `i` based on the
 * messages that precede it (role + textual content). This lets us look up
 * the original thinking blocks we cached when that assistant reply was
 * produced.
 */
export function assistantTurnFingerprint(
  messages: AnthropicMessage[],
  assistantIndex: number,
  model: string | undefined,
): string {
  const prefix = messages.slice(0, assistantIndex).map((m) => ({
    role: m.role,
    content: normalizeContentForFingerprint(m.content),
  }));
  return fingerprint({ model: model ?? "", prefix });
}

function normalizeContentForFingerprint(
  content: AnthropicMessage["content"],
): unknown {
  if (typeof content === "string") return content;
  // Drop signatures / volatile metadata; keep structural identity.
  return content.map((c) => {
    if (c.type === "text") return { type: "text", text: (c as any).text };
    if (c.type === "tool_use")
      return {
        type: "tool_use",
        id: (c as any).id,
        name: (c as any).name,
        input: (c as any).input,
      };
    if (c.type === "tool_result")
      return {
        type: "tool_result",
        tool_use_id: (c as any).tool_use_id,
        content: (c as any).content,
      };
    if (c.type === "thinking" || c.type === "redacted_thinking") {
      // thinking blocks themselves should not affect the fingerprint,
      // because we re-inject them based on the non-thinking prefix.
      return { type: c.type };
    }
    return { type: c.type };
  });
}

/**
 * For each assistant message in the request that is missing thinking blocks,
 * look the cached blocks up and prepend them to `content`.
 *
 * Returns true if any message was mutated.
 */
export function reinjectThinkingBlocks(
  body: AnthropicRequestBody,
  lookup: (fp: string) => ThinkingBlock[] | undefined,
): boolean {
  if (!body?.messages?.length) return false;
  let mutated = false;

  for (let i = 0; i < body.messages.length; i++) {
    const m = body.messages[i]!;
    if (m.role !== "assistant") continue;
    if (typeof m.content === "string") {
      // Normalize string content into blocks so we can prepend.
      m.content = m.content ? [{ type: "text", text: m.content }] : [];
    }
    if (hasThinkingBlock(m)) continue;

    const fp = assistantTurnFingerprint(body.messages, i, body.model);
    const blocks = lookup(fp);
    if (!blocks?.length) continue;

    // Prepend the cached thinking blocks. They must be at the start of the
    // assistant content per Anthropic's extended-thinking contract.
    (m.content as AnthropicContentBlock[]) = [
      ...(blocks as AnthropicContentBlock[]),
      ...(m.content as AnthropicContentBlock[]),
    ];
    mutated = true;
  }

  return mutated;
}

export interface AnthropicPlaceholderOptions {
  /** off: never inject. fallback: only when missing. always: unconditionally prepend. */
  mode: "off" | "fallback" | "always";
  /** Text used for the placeholder thinking block. Must be non-empty. */
  text: string;
  /**
   * How to handle the `signature` field on the placeholder block.
   * - "empty" (default): emit `signature: ""` — works for DeepSeek-V4
   *   Anthropic-shape relays that don't validate signatures.
   * - "omit": don't emit a `signature` field at all.
   */
  signaturePolicy?: "empty" | "omit";
}

/**
 * Fill historical assistant messages with a placeholder `thinking` block when
 * they are missing one. Mirrors `fillReasoningPlaceholder` on the OpenAI side.
 *
 * NOTE: real Anthropic validates `signature` cryptographically. This helper is
 * intended for DeepSeek-V4 / relay endpoints that mimic the Anthropic shape
 * but do not enforce the signature contract. Returns the number of mutated
 * assistant messages.
 *
 * Call this AFTER `reinjectThinkingBlocks` so cached real thinking wins.
 */
export function fillThinkingPlaceholder(
  body: AnthropicRequestBody,
  opts: AnthropicPlaceholderOptions,
): number {
  if (opts.mode === "off") return 0;
  if (!body?.messages?.length) return 0;
  const text =
    opts.text && opts.text.length > 0 ? opts.text : "(thinking omitted)";
  const policy = opts.signaturePolicy ?? "empty";
  let mutated = 0;

  for (const m of body.messages) {
    if (m.role !== "assistant") continue;
    if (typeof m.content === "string") {
      m.content = m.content ? [{ type: "text", text: m.content }] : [];
    }
    if (!Array.isArray(m.content)) continue;
    if (opts.mode === "fallback" && hasThinkingBlock(m)) continue;

    const block: AnthropicContentBlock =
      policy === "empty"
        ? { type: "thinking", thinking: text, signature: "" }
        : { type: "thinking", thinking: text };
    (m.content as AnthropicContentBlock[]).unshift(block);
    mutated++;
  }
  return mutated;
}

/**
 * Extract thinking blocks from a non-streamed response body.
 */
export function extractThinkingFromResponse(
  json: unknown,
): ThinkingBlock[] {
  if (!json || typeof json !== "object") return [];
  const content = (json as any).content;
  if (!Array.isArray(content)) return [];
  return content.filter(
    (c: any) => c && (c.type === "thinking" || c.type === "redacted_thinking"),
  ) as ThinkingBlock[];
}
