import type { ThinkingBlock } from "./cache.js";

/**
 * Tee a Response body, while parsing the Anthropic Messages SSE stream on the
 * fly to reconstruct the thinking blocks returned by the server. The original
 * body is passed through untouched to the caller.
 *
 * Anthropic streaming event shape (abridged):
 *   event: content_block_start
 *   data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}
 *
 *   event: content_block_delta
 *   data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"..."}}
 *
 *   event: content_block_delta
 *   data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"..."}}
 *
 *   event: content_block_stop
 *   data: {"type":"content_block_stop","index":0}
 */
export function interceptStreamForThinking(
  upstream: Response,
  onThinkingBlocks: (blocks: ThinkingBlock[]) => void,
): Response {
  if (!upstream.body) return upstream;

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();

  // index -> partial block being assembled
  const blocksByIndex = new Map<number, ThinkingBlock & { _finalized?: boolean }>();
  let buffer = "";

  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (done) {
          flushBlocks();
          controller.close();
          return;
        }
        // pass-through
        controller.enqueue(value);

        // parse for thinking extraction
        buffer += decoder.decode(value, { stream: true });
        let nl: number;
        // Anthropic SSE events are separated by blank lines (\n\n)
        while ((nl = buffer.indexOf("\n\n")) !== -1) {
          const rawEvent = buffer.slice(0, nl);
          buffer = buffer.slice(nl + 2);
          handleEvent(rawEvent);
        }
      } catch (err) {
        controller.error(err);
      }
    },
    cancel(reason) {
      reader.cancel(reason).catch(() => {});
    },
  });

  function handleEvent(raw: string): void {
    // An SSE event is a set of `field: value` lines; we only care about data
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
    if (!ev || typeof ev !== "object") return;

    switch (ev.type) {
      case "content_block_start": {
        const idx = ev.index as number;
        const cb = ev.content_block;
        if (cb && (cb.type === "thinking" || cb.type === "redacted_thinking")) {
          blocksByIndex.set(idx, {
            type: cb.type,
            thinking: cb.thinking ?? "",
            data: cb.data,
            signature: cb.signature,
          });
        }
        break;
      }
      case "content_block_delta": {
        const idx = ev.index as number;
        const block = blocksByIndex.get(idx);
        if (!block) return;
        const d = ev.delta ?? {};
        if (d.type === "thinking_delta" && typeof d.thinking === "string") {
          block.thinking = (block.thinking ?? "") + d.thinking;
        } else if (d.type === "signature_delta" && typeof d.signature === "string") {
          block.signature = (block.signature ?? "") + d.signature;
        } else if (d.type === "input_json_delta") {
          // not relevant
        }
        break;
      }
      case "content_block_stop": {
        const idx = ev.index as number;
        const block = blocksByIndex.get(idx);
        if (block) block._finalized = true;
        break;
      }
      case "message_stop": {
        flushBlocks();
        break;
      }
      default:
        break;
    }
  }

  let flushed = false;
  function flushBlocks(): void {
    if (flushed) return;
    flushed = true;
    const blocks: ThinkingBlock[] = [];
    const indices = [...blocksByIndex.keys()].sort((a, b) => a - b);
    for (const i of indices) {
      const b = blocksByIndex.get(i)!;
      const { _finalized: _f, ...rest } = b;
      blocks.push(rest);
    }
    if (blocks.length) onThinkingBlocks(blocks);
  }

  // unused reference to encoder to keep import side-effect-free-safe on strict compilers
  void encoder;

  return new Response(stream, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: upstream.headers,
  });
}
