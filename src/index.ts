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
  fillReasoningPlaceholder,
  type OpenAIChatRequestBody,
  type OpenAIReasoningPayload,
  type PlaceholderOptions,
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
  /**
   * Placeholder reasoning injection for OpenAI-compatible requests.
   *
   * Useful for rescuing old conversations whose real thinking content was
   * never captured. Only applied when protocol === "openai" — Anthropic
   * thinking blocks use cryptographic signatures that can't be faked.
   *
   * - mode "off"       : never inject placeholders
   * - mode "fallback"  : only when cache miss left an assistant without reasoning (default)
   * - mode "always"    : fill every historical assistant turn regardless of cache
   *
   * Default: { mode: "fallback", text: "(thinking omitted)" }.
   */
  placeholder?: Partial<PlaceholderOptions>;
  /**
   * When true, the wrapper only acts on requests whose body has a `model`
   * field containing a deepseek keyword. Non-matching requests are passed
   * through unchanged.
   *
   * This is set automatically when a provider was wrapped because of its
   * model list (rather than its id) so that sibling models on the same
   * provider (e.g. glm, gemini) are not touched.
   *
   * Default: false (act on every request the wrapper sees).
   */
  requireModelKeywordMatch?: boolean;
}

type FetchFn = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>;

type Protocol = "anthropic" | "openai" | "unknown";

const DEEPSEEK_KEYWORDS = ["deepseek", "deep-seek", "ds-v4", "dsv4"];

function matchesDeepSeekKeyword(s: string): boolean {
  if (!s) return false;
  const lower = s.toLowerCase();
  return DEEPSEEK_KEYWORDS.some((k) => lower.includes(k));
}

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
function wrapFetchForDeepSeekThinking(
  upstreamFetch: FetchFn = globalThis.fetch,
  opts: DeepSeekThinkingFixOptions = {},
): FetchFn {
  const {
    ttlMs = 30 * 60 * 1000,
    debug = false,
    ensureThinkingEnabled = true,
    defaultBudgetTokens = 8000,
    handleOpenAI = true,
    requireModelKeywordMatch = false,
  } = opts;
  const placeholderOpts: PlaceholderOptions = {
    mode: opts.placeholder?.mode ?? "fallback",
    // Use a recognizable non-whitespace string so that relays which trim
    // payloads can't accidentally collapse the placeholder back to empty.
    text: opts.placeholder?.text ?? "(thinking omitted)",
    field: opts.placeholder?.field ?? "reasoning_content",
  };

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
        if (debug)
          console.log(
            "[deepseek-thinking-fix] passthrough: body is not string",
            { hasInit: !!init, bodyType: typeof init?.body },
          );
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
        if (debug)
          console.log("[deepseek-thinking-fix] passthrough: body not JSON", url);
        return upstreamFetch(input, init);
      }
      if (!body || !Array.isArray(body.messages)) {
        if (debug)
          console.log(
            "[deepseek-thinking-fix] passthrough: no messages array",
            { url, keys: body ? Object.keys(body) : null },
          );
        return upstreamFetch(input, init);
      }

      if (requireModelKeywordMatch) {
        const m = typeof body.model === "string" ? body.model : "";
        if (!matchesDeepSeekKeyword(m)) {
          if (debug)
            console.log(
              `[deepseek-thinking-fix] passthrough: model="${m}" does not match deepseek keyword`,
            );
          return upstreamFetch(input, init);
        }
      }

      const protocol = detectProtocol(url, body);
      if (debug) {
        console.log(
          `[deepseek-thinking-fix] intercept url=${url} protocol=${protocol} msgs=${body.messages.length}`,
        );
      }
      if (protocol === "anthropic") {
        return handleAnthropic(url, init, body as AnthropicRequestBody);
      }
      if (protocol === "openai" && handleOpenAI) {
        return handleOpenAIReq(url, init, body as OpenAIChatRequestBody);
      }
      if (debug)
        console.log(
          `[deepseek-thinking-fix] passthrough: protocol=${protocol} handleOpenAI=${handleOpenAI}`,
        );
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
    if (debug) {
      const summary = (body.messages ?? []).map((m, i) => ({
        i,
        role: m.role,
        contentType: Array.isArray(m.content)
          ? `array[${m.content.length}]`
          : typeof m.content,
        contentLen:
          typeof m.content === "string"
            ? m.content.length
            : Array.isArray(m.content)
              ? m.content.length
              : 0,
        hasReasoningContent:
          typeof (m as any).reasoning_content === "string" &&
          (m as any).reasoning_content.length > 0,
        hasReasoning:
          typeof (m as any).reasoning === "string" &&
          (m as any).reasoning.length > 0,
        toolCalls: Array.isArray((m as any).tool_calls)
          ? (m as any).tool_calls.length
          : 0,
        toolCallId: (m as any).tool_call_id,
      }));
      console.log(
        "[deepseek-thinking-fix][openai] inbound messages:",
        JSON.stringify(summary),
      );
    }

    const mutated = reinjectReasoningContent(body, (fp) => oaGet(fp));
    if (debug) {
      console.log(
        `[deepseek-thinking-fix][openai] cache reinject mutated=${mutated}`,
      );
    }

    if (placeholderOpts.mode !== "off") {
      const filled = fillReasoningPlaceholder(body, placeholderOpts);
      if (debug) {
        console.log(
          `[deepseek-thinking-fix][openai] placeholder filled ${filled} assistant message(s) (mode=${placeholderOpts.mode}, text="${placeholderOpts.text}", field=${placeholderOpts.field})`,
        );
      }
    } else if (debug) {
      console.log(
        "[deepseek-thinking-fix][openai] placeholder skipped (mode=off)",
      );
    }

    if (debug) {
      const after = (body.messages ?? []).map((m, i) => ({
        i,
        role: m.role,
        rcLen:
          typeof (m as any).reasoning_content === "string"
            ? (m as any).reasoning_content.length
            : null,
        rLen:
          typeof (m as any).reasoning === "string"
            ? (m as any).reasoning.length
            : null,
      }));
      console.log(
        "[deepseek-thinking-fix][openai] outbound messages reasoning:",
        JSON.stringify(after),
      );
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
  /**
   * A provider qualifies if either its id or **any of its model ids** contains
   * a deepseek keyword. The latter case is common when users alias DeepSeek
   * behind a custom relay name (e.g. provider id="lynnaiv4" hosting a
   * "deepseek-v4-pro" model). The fetch wrapper still re-checks `body.model`
   * at request time so non-deepseek models on the same provider are passed
   * through untouched.
   */
  const providerMatches = (id: string, p: any): boolean => {
    if (matchesDeepSeekKeyword(id)) return true;
    const models = p?.models;
    if (models && typeof models === "object") {
      for (const mid of Object.keys(models)) {
        if (matchesDeepSeekKeyword(mid)) return true;
      }
    }
    return false;
  };

  return {
    config: async (config: any) => {
      const providers = config?.provider ?? {};
      const allIds = Object.keys(providers);
      const matched: string[] = [];
      for (const [id, p] of Object.entries(providers) as [string, any][]) {
        if (!providerMatches(id, p)) continue;
        matched.push(id);
        p.options ??= {};
        const originalFetch: FetchFn | undefined = p.options.fetch;
        const dbg = !!p.options?.debugThinking;
        const idMatches = matchesDeepSeekKeyword(id);
        p.options.fetch = wrapFetchForDeepSeekThinking(
          originalFetch ?? ((globalThis as any).fetch as FetchFn),
          {
            debug: dbg,
            placeholder: p.options?.thinkingPlaceholder,
            ttlMs: p.options?.thinkingTtlMs,
            ensureThinkingEnabled: p.options?.ensureThinkingEnabled,
            defaultBudgetTokens: p.options?.defaultBudgetTokens,
            handleOpenAI: p.options?.handleOpenAI,
            // When the provider id itself does not contain a deepseek
            // keyword, only act when the request body's model field does.
            requireModelKeywordMatch: !idMatches,
          },
        );
        if (dbg) {
          console.log(
            `[deepseek-thinking-fix] wrapped fetch on provider id="${id}" (originalFetch=${!!originalFetch}, requireModelKeywordMatch=${!idMatches})`,
          );
        }
      }
      const anyDebug = Object.values(providers).some(
        (p: any) => !!p?.options?.debugThinking,
      );
      if (anyDebug) {
        console.log(
          `[deepseek-thinking-fix] config hook ran. all providers=[${allIds.join(",")}] matched=[${matched.join(",")}]`,
        );
      }
    },

    "chat.params": async (
      input: { model?: { id?: string }; provider?: { id?: string } },
      output: { options?: Record<string, unknown> },
    ) => {
      const pid = input?.provider?.id ?? "";
      const mid = input?.model?.id ?? "";
      // Match by provider id OR model id, mirroring the config-time logic.
      if (
        !matchesDeepSeekKeyword(pid) &&
        !matchesDeepSeekKeyword(mid)
      )
        return;
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

// Note: only the default export is intended for opencode's plugin loader.
// Programmatic helpers and types live in `./api.js` to avoid the loader
// trying to invoke a class constructor as a plugin.
export type { ThinkingBlock, AnthropicRequestBody };
export type {
  OpenAIChatRequestBody,
  OpenAIReasoningPayload,
  PlaceholderOptions,
} from "./messages-openai.js";
