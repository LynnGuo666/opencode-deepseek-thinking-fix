# opencode-deepseek-thinking-fix

OpenCode plugin that fixes the following DeepSeek-V4 (Anthropic-compatible) error:

```
The `content[].thinking` in the thinking mode must be passed back to the API.
```

DeepSeek-V4 follows Anthropic's extended-thinking contract: when `thinking` is
enabled, every assistant message you replay in the next turn **must include the
original thinking blocks verbatim (with their `signature` field)**. OpenCode and
several upstream libraries strip or reorder these blocks, which makes the
server reject the request.

Works for **both protocols**:

- Anthropic-compatible (`/v1/messages`) — thinking blocks + `signature`
- OpenAI-compatible (`/v1/chat/completions`) — `reasoning_content` / `reasoning`

Detection is automatic based on URL and body shape.

This plugin wraps the provider's `fetch` so that:

1. On every response, the original thinking blocks are **captured** (both in
   streaming and non-streaming mode).
2. On every subsequent request, assistant messages that are missing thinking
   blocks have them **re-injected** from cache (keyed by the fingerprint of the
   preceding messages).
3. The `thinking` parameter is **force-enabled** on DeepSeek requests so the
   server keeps producing them.

## Install

Global:

```bash
cd ~/.config/opencode
bun add opencode-deepseek-thinking-fix
```

Then edit `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["opencode-deepseek-thinking-fix"]
}
```

Or load locally while developing:

```json
{
  "plugin": [""]
}
```

## How it works

- Registers a `config` hook that finds any provider whose id contains
  `deepseek` (configurable) and replaces `provider.options.fetch` with a
  wrapped version.
- The wrapped fetch inspects JSON bodies; if it looks like an Anthropic
  `/messages` request it:
  - ensures `thinking: { type: "enabled", budget_tokens: 8000 }` is present,
  - computes a fingerprint of every assistant-message prefix,
  - for each assistant message missing thinking blocks, prepends the cached
    ones.
- After the response is received:
  - non-stream: clones the body, parses JSON, stores thinking blocks.
  - stream (SSE): tees the body, parses `content_block_start / delta / stop`
    events to reconstruct the thinking blocks (including assembling
    `signature_delta` chunks), then stores them.

## Options

You can also use the lower-level helper directly if you embed OpenCode:

```ts
import { wrapFetchForDeepSeekThinking } from "opencode-deepseek-thinking-fix";

const fetchWithFix = wrapFetchForDeepSeekThinking(globalThis.fetch, {
  debug: true,
  ttlMs: 30 * 60 * 1000,
  ensureThinkingEnabled: true,
  defaultBudgetTokens: 8000,
});
```

## Build

```bash
bun install
bun run build
```

## License

MIT
