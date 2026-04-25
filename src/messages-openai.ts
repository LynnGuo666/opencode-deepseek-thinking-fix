import { fingerprint } from "./cache.js";

export interface OpenAIMessage {
  role: "system" | "user" | "assistant" | "tool" | "developer" | string;
  content?: string | OpenAIContentPart[] | null;
  reasoning_content?: string | null;
  reasoning?: string | null;
  tool_calls?: unknown[];
  tool_call_id?: string;
  name?: string;
  [k: string]: unknown;
}

export type OpenAIContentPart =
  | { type: "text"; text: string; [k: string]: unknown }
  | { type: "image_url"; [k: string]: unknown }
  | { type: string; [k: string]: unknown };

export interface OpenAIChatRequestBody {
  model?: string;
  messages?: OpenAIMessage[];
  stream?: boolean;
  reasoning_effort?: string;
  [k: string]: unknown;
}

export interface OpenAIReasoningPayload {
  /** The `reasoning_content` string (DeepSeek / official name). */
  reasoning_content?: string;
  /** Some forks use `reasoning` instead. */
  reasoning?: string;
}

export function hasReasoning(msg: OpenAIMessage): boolean {
  return (
    (typeof msg.reasoning_content === "string" &&
      msg.reasoning_content.length > 0) ||
    (typeof msg.reasoning === "string" && msg.reasoning.length > 0)
  );
}

function normalizeContentForFingerprint(content: OpenAIMessage["content"]): unknown {
  if (content == null) return null;
  if (typeof content === "string") return content;
  return content.map((c) => {
    if (c.type === "text") return { type: "text", text: (c as any).text };
    return { type: c.type };
  });
}

export function assistantTurnFingerprintOpenAI(
  messages: OpenAIMessage[],
  assistantIndex: number,
  model: string | undefined,
): string {
  const prefix = messages.slice(0, assistantIndex).map((m) => ({
    role: m.role,
    content: normalizeContentForFingerprint(m.content ?? null),
    tool_call_id: m.tool_call_id,
    tool_calls: m.tool_calls?.map((tc: any) => ({
      id: tc?.id,
      type: tc?.type,
      function: tc?.function
        ? { name: tc.function.name, arguments: tc.function.arguments }
        : undefined,
    })),
    name: m.name,
  }));
  return fingerprint({ model: model ?? "", prefix, protocol: "openai" });
}

/**
 * For each assistant message in the request missing `reasoning_content`,
 * look the cached payload up and attach it back.
 */
export function reinjectReasoningContent(
  body: OpenAIChatRequestBody,
  lookup: (fp: string) => OpenAIReasoningPayload | undefined,
): boolean {
  if (!body?.messages?.length) return false;
  let mutated = false;
  for (let i = 0; i < body.messages.length; i++) {
    const m = body.messages[i]!;
    if (m.role !== "assistant") continue;
    if (hasReasoning(m)) continue;
    const fp = assistantTurnFingerprintOpenAI(body.messages, i, body.model);
    const payload = lookup(fp);
    if (!payload) continue;
    if (payload.reasoning_content) m.reasoning_content = payload.reasoning_content;
    if (payload.reasoning) m.reasoning = payload.reasoning;
    mutated = true;
  }
  return mutated;
}

export function extractReasoningFromOpenAIResponse(
  json: unknown,
): OpenAIReasoningPayload | undefined {
  if (!json || typeof json !== "object") return;
  const choice = (json as any).choices?.[0];
  const msg = choice?.message;
  if (!msg) return;
  const out: OpenAIReasoningPayload = {};
  if (typeof msg.reasoning_content === "string" && msg.reasoning_content)
    out.reasoning_content = msg.reasoning_content;
  if (typeof msg.reasoning === "string" && msg.reasoning)
    out.reasoning = msg.reasoning;
  if (!out.reasoning_content && !out.reasoning) return;
  return out;
}

export function stripReasoningForFingerprint(msg: OpenAIMessage): OpenAIMessage {
  const { reasoning_content: _a, reasoning: _b, ...rest } = msg;
  return rest as OpenAIMessage;
}

/**
 * Decide whether a given assistant message needs reasoning content to be
 * replayed.
 *
 * Aggressive strategy: any `role === "assistant"` qualifies. DeepSeek's
 * server-side check only verifies that `reasoning_content` exists and is
 * non-empty — over-filling never corrupts prefill or attention because the
 * model conditions on `content`, not on the reasoning field. Filtering by
 * content/tool_calls historically dropped legitimate but minimal assistant
 * turns (`content: ""`, `content: null`, pure-thinking turns) and caused
 * rescue to silently miss them.
 */
export function isHistoricalAssistantMessage(msg: OpenAIMessage): boolean {
  return msg.role === "assistant";
}

export interface PlaceholderOptions {
  /** off: never inject. fallback: only when cache missed. always: unconditionally. */
  mode: "off" | "fallback" | "always";
  /** Text used for the placeholder. Must be non-empty. */
  text: string;
  /** Which field to fill. Default: "reasoning_content". */
  field?: "reasoning_content" | "reasoning" | "both";
}

/**
 * Fill `reasoning_content` / `reasoning` on assistant history messages that
 * still don't have any. Returns number of messages mutated.
 *
 * Call this AFTER `reinjectReasoningContent` so cached real thinking wins.
 */
export function fillReasoningPlaceholder(
  body: OpenAIChatRequestBody,
  opts: PlaceholderOptions,
): number {
  if (opts.mode === "off") return 0;
  if (!body?.messages?.length) return 0;
  const text =
    opts.text && opts.text.length > 0 ? opts.text : "(thinking omitted)";
  const field = opts.field ?? "reasoning_content";
  let mutated = 0;

  for (const m of body.messages) {
    if (!isHistoricalAssistantMessage(m)) continue;
    if (opts.mode === "fallback" && hasReasoning(m)) continue;
    if (field === "reasoning_content" || field === "both") {
      if (!m.reasoning_content) {
        m.reasoning_content = text;
        mutated++;
      }
    }
    if (field === "reasoning" || field === "both") {
      if (!m.reasoning) {
        m.reasoning = text;
        mutated++;
      }
    }
  }
  return mutated;
}
