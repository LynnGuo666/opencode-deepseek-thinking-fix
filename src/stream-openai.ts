import type { OpenAIReasoningPayload } from "./messages-openai.js";

/**
 * Tee an OpenAI-style Chat Completions SSE response, accumulating
 * `reasoning_content` / `reasoning` deltas so we can cache them for replay
 * in the next turn.
 *
 * Event shape (DeepSeek / OpenAI-compat):
 *   data: {"id":"...","choices":[{"delta":{"reasoning_content":"..."},"index":0}]}
 *   data: {"id":"...","choices":[{"delta":{"content":"..."},"index":0}]}
 *   data: [DONE]
 */
export function interceptOpenAIStreamForReasoning(
  upstream: Response,
  onFinal: (payload: OpenAIReasoningPayload) => void,
): Response {
  if (!upstream.body) return upstream;
  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";
  let reasoningContent = "";
  let reasoning = "";

  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (done) {
          flush();
          controller.close();
          return;
        }
        controller.enqueue(value);
        buffer += decoder.decode(value, { stream: true });
        let nl: number;
        while ((nl = buffer.indexOf("\n\n")) !== -1) {
          const raw = buffer.slice(0, nl);
          buffer = buffer.slice(nl + 2);
          handle(raw);
        }
      } catch (err) {
        controller.error(err);
      }
    },
    cancel(reason) {
      reader.cancel(reason).catch(() => {});
    },
  });

  function handle(raw: string): void {
    const dataLines: string[] = [];
    for (const line of raw.split("\n")) {
      if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return;
    const payload = dataLines.join("\n");
    if (!payload || payload === "[DONE]") return;

    let ev: any;
    try {
      ev = JSON.parse(payload);
    } catch {
      return;
    }
    const delta = ev?.choices?.[0]?.delta;
    if (!delta) return;
    if (typeof delta.reasoning_content === "string") {
      reasoningContent += delta.reasoning_content;
    }
    if (typeof delta.reasoning === "string") {
      reasoning += delta.reasoning;
    }
  }

  let flushed = false;
  function flush(): void {
    if (flushed) return;
    flushed = true;
    const out: OpenAIReasoningPayload = {};
    if (reasoningContent) out.reasoning_content = reasoningContent;
    if (reasoning) out.reasoning = reasoning;
    if (out.reasoning_content || out.reasoning) onFinal(out);
  }

  return new Response(stream, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: upstream.headers,
  });
}
