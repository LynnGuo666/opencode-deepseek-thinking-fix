import { ThinkingCache, fingerprint, type ThinkingBlock } from "./cache.js";
import {
  reinjectThinkingBlocks,
  extractThinkingFromResponse,
  assistantTurnFingerprint,
  type AnthropicRequestBody,
} from "./messages.js";
import { interceptStreamForThinking } from "./stream.js";
import {
  reinjectReasoningContent,
  extractReasoningFromOpenAIResponse,
  assistantTurnFingerprintOpenAI,
  type OpenAIChatRequestBody,
  type OpenAIReasoningPayload,
} from "./messages-openai.js";
import { interceptOpenAIStreamForReasoning } from "./stream-openai.js";

export interface DeepSeekThinkingFixOptions {
  /** Provider ids to wrap (substring match). Default: ["deepseek"] */
  providerIds?: string[];
  /** TTL for cached thinking blocks (ms). Default: 30 minutes. */
  ttlMs?: number;
  /** Extra logging. Default: false. */
  debug?: boolean;
  /**
   * Ensure the `thinking` parameter is present in Anthropic-shape requests
   * even if the caller forgot to send it. Default: true.
   */
  ensureThinkingEnabled?: boolean;
  /** Budget tokens used when `ensureThinkingEnabled` injects the param. */
  defaultBudgetTokens?: number;
  /**
   * When true, also handle OpenAI-compatible (/chat/completions) bodies.
   * Default: true.
   */
  handleOpenAI?: boolean;
}

type FetchFn = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>;

type Protocol = "anthropic" | "openai" | "unknown";

function detectProtocol(url: string, body: any): Protocol {
  const u = url.toLowerCase();
  if (u.includes("/v1/messages") || u.endsWith("/messages")) return "anthropic";
  if (u.includes("/chat/completions")) return "openai";

  // Fallback: inspect body shape.
  if (!body || typeof body !== "object") return "unknown";
  if (Array.isArray(body.messages)) {
    // Anthropic requests carry top-level `system` string/array and usually
    // content blocks. OpenAI uses role:"system" inside messages.
    const hasSystemField =
      typeof body.system === "string" || Array.isArray(body.system);
    if (hasSystemField) return "anthropic";
    const hasSystemRole = body.messages.some(
      (m: any) => m?.role === "system" || m?.role === "developer",
    );
    if (hasSystemRole) return "openai";
    // thinking param is Anthropic; reasoning_effort is OpenAI.
    if (body.thinking) return "anthropic";
    if (body.reasoning_effort) return "openai";
  }
  return "unknown";
}

/**
 * Wrap a fetch function so that requests to DeepSeek-V4 (or other Anthropic-
 * and OpenAI-compatible endpoints) preserve the thinking/reasoning content
 * across turns.
 */
export function wrapFetchForDeepSeekThinking(
  upstreamFetch: FetchFn = globalThis.fetch,
  opts: DeepSeekThinkingFixOptions = {},
): FetchFn {
  const {
    ttlMs = 30 * 60 * 1000,
    debug = false,
    ensureThinkingEnabled = true,
    defaultBudgetTokens = 8000,
    handleOpenAI = true,
  } = opts;

  const anthCache = new ThinkingCache(ttlMs);
  const oaCache = new Map<string, { payload: OpenAIReasoningPayload; at: number }>();

  function oaGet(fp: string): OpenAIReasoningPayload | undefined {
    const e = oaCache.get(fp);
    if (!e) return;
    if (Date.now() - e.at > ttlMs) {
      oaCache.delete(fp);
      return;
    }
    return e.payload;
  }
  function oaSet(fp: string, payload: OpenAIReasoningPayload): void {
    if (!payload.reasoning_content && !payload.reasoning) return;
    if (oaCache.size > 500) {
      const oldest = [...oaCache.entries()].sort((a, b) => a[1].at - b[1].at)[0];
      if (oldest) oaCache.delete(oldest[0]);
    }
    oaCache.set(fp, { payload, at: Date.now() });
  }

  return async function wrappedFetch(input, init) {
    try {
      if (!init?.body || typeof init.body !== "string") {
        return upstreamFetch(input, init);
      }
      const url =
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.toString()
            : (input as Request).url;

      let body: any;
      try {
        body = JSON.parse(init.body);
      } catch {
        return upstreamFetch(input, init);
      }
      if (!body || !Array.isArray(body.messages)) {
        return upstreamFetch(input, init);
      }

      const protocol = detectProtocol(url, body);
      if (protocol === "anthropic") {
        return handleAnthropic(url, init, body as AnthropicRequestBody);
      }
      if (protocol === "openai" && handleOpenAI) {
        return handleOpenAIReq(url, init, body as OpenAIChatRequestBody);
      }
      return upstreamFetch(input, init);
    } catch (err) {
      if (debug) console.error("[deepseek-thinking-fix] error:", err);
      return upstreamFetch(input, init);
    }
  };

  async function handleAnthropic(
    url: string,
    init: RequestInit,
    body: AnthropicRequestBody,
  ): Promise<Response> {
    if (ensureThinkingEnabled && body.thinking == null) {
      body.thinking = { type: "enabled", budget_tokens: defaultBudgetTokens };
    }
    const mutated = reinjectThinkingBlocks(body, (fp) => anthCache.get(fp));
    if (mutated && debug) {
      console.log("[deepseek-thinking-fix][anthropic] re-injected thinking blocks");
    }

    const newInit: RequestInit = { ...init, body: JSON.stringify(body) };
    const response = await upstreamFetch(url, newInit);

    const nextFp = fingerprint({
      model: body.model ?? "",
      prefix: (body.messages ?? []).map((m) => ({
        role: m.role,
        content: stripAnthropicThinking(m.content),
      })),
      protocol: "anthropic",
    });

    const ct = response.headers.get("content-type") ?? "";
    const isStream = ct.includes("text/event-stream") || body.stream === true;
    if (!response.ok) return response;

    if (isStream) {
      return interceptStreamForThinking(response, (blocks) => {
        if (blocks.length) {
          anthCache.set(nextFp, blocks);
          if (debug)
            console.log(
              `[deepseek-thinking-fix][anthropic] cached ${blocks.length} thinking blocks (stream)`,
            );
        }
      });
    }

    response
      .clone()
      .json()
      .then((json) => {
        const blocks = extractThinkingFromResponse(json);
        if (blocks.length) {
          anthCache.set(nextFp, blocks);
          if (debug)
            console.log(
              `[deepseek-thinking-fix][anthropic] cached ${blocks.length} thinking blocks (json)`,
            );
        }
      })
      .catch(() => {});
    return response;
  }

  async function handleOpenAIReq(
    url: string,
    init: RequestInit,
    body: OpenAIChatRequestBody,
  ): Promise<Response> {
    const mutated = reinjectReasoningContent(body, (fp) => oaGet(fp));
    if (mutated && debug) {
      console.log("[deepseek-thinking-fix][openai] re-injected reasoning_content");
    }

    const newInit: RequestInit = { ...init, body: JSON.stringify(body) };
    const response = await upstreamFetch(url, newInit);

    const nextFp = assistantTurnFingerprintOpenAI(
      // The "next" assistant turn will be at messages.length (after the
      // current user input). Use body.messages as the full prefix.
      [...(body.messages ?? []), { role: "assistant" }],
      body.messages?.length ?? 0,
      body.model,
    );

    const ct = response.headers.get("content-type") ?? "";
    const isStream = ct.includes("text/event-stream") || body.stream === true;
    if (!response.ok) return response;

    if (isStream) {
      return interceptOpenAIStreamForReasoning(response, (payload) => {
        oaSet(nextFp, payload);
        if (debug)
          console.log("[deepseek-thinking-fix][openai] cached reasoning (stream)");
      });
    }

    response
      .clone()
      .json()
      .then((json) => {
        const payload = extractReasoningFromOpenAIResponse(json);
        if (payload) {
          oaSet(nextFp, payload);
          if (debug)
            console.log("[deepseek-thinking-fix][openai] cached reasoning (json)");
        }
      })
      .catch(() => {});
    return response;
  }
}

function stripAnthropicThinking(content: unknown): unknown {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return content;
  return content
    .filter(
      (c: any) =>
        c && c.type !== "thinking" && c.type !== "redacted_thinking",
    )
    .map((c: any) => {
      if (c.type === "text") return { type: "text", text: c.text };
      if (c.type === "tool_use")
        return { type: "tool_use", id: c.id, name: c.name, input: c.input };
      if (c.type === "tool_result")
        return {
          type: "tool_result",
          tool_use_id: c.tool_use_id,
          content: c.content,
        };
      return { type: c.type };
    });
}

/**
 * OpenCode plugin entry. Wraps matching providers' fetch so DeepSeek-V4
 * (both Anthropic- and OpenAI-compatible protocols) preserves thinking
 * content across turns.
 */
const DeepSeekThinkingFix = (async (_ctx: unknown) => {
  const providerMatches = (id: string) =>
    ["deepseek", "deep-seek", "ds-v4", "dsv4"].some((s) =>
      id.toLowerCase().includes(s),
    );

  return {
    config: async (config: any) => {
      const providers = config?.provider ?? {};
      for (const [id, p] of Object.entries(providers) as [string, any][]) {
        if (!providerMatches(id)) continue;
        p.options ??= {};
        const originalFetch: FetchFn | undefined = p.options.fetch;
        p.options.fetch = wrapFetchForDeepSeekThinking(
          originalFetch ?? ((globalThis as any).fetch as FetchFn),
          { debug: !!p.options?.debugThinking },
        );
      }
    },

    "chat.params": async (
      input: { model?: { id?: string }; provider?: { id?: string } },
      output: { options?: Record<string, unknown> },
    ) => {
      const pid = input?.provider?.id ?? "";
      if (!providerMatches(pid)) return;
      output.options ??= {};
      if ((output.options as any).thinking == null) {
        (output.options as any).thinking = {
          type: "enabled",
          budget_tokens: 8000,
        };
      }
    },
  };
}) as unknown as (ctx: unknown) => Promise<Record<string, unknown>>;

export default DeepSeekThinkingFix;
export { DeepSeekThinkingFix };
export { ThinkingCache, fingerprint, assistantTurnFingerprint };
export type { ThinkingBlock, AnthropicRequestBody };
export type { OpenAIChatRequestBody, OpenAIReasoningPayload };
