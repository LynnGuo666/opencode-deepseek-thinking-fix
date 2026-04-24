# opencode-deepseek-thinking-fix

[![npm version](https://img.shields.io/npm/v/opencode-deepseek-thinking-fix.svg)](https://www.npmjs.com/package/opencode-deepseek-thinking-fix)
[![npm downloads](https://img.shields.io/npm/dm/opencode-deepseek-thinking-fix.svg)](https://www.npmjs.com/package/opencode-deepseek-thinking-fix)
[![license](https://img.shields.io/npm/l/opencode-deepseek-thinking-fix.svg)](./LICENSE)

OpenCode plugin that preserves DeepSeek-V4 (and other reasoning models') thinking content across turns, fixing errors like:

```
The `content[].thinking` in the thinking mode must be passed back to the API.
```

```
messages.X.content.0.type: Expected 'thinking' or 'redacted_thinking', but found 'tool_use'
```

Supports **both** upstream protocols. Detection is automatic.

| Protocol | Endpoint | What's preserved |
|---|---|---|
| Anthropic-compatible | `/v1/messages` | `content[]` thinking / redacted_thinking blocks with `signature` |
| OpenAI-compatible | `/v1/chat/completions` | `reasoning_content` (and `reasoning` fallback) |

---

## Install

One line in your opencode config:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["opencode-deepseek-thinking-fix"]
}
```

OpenCode will auto-install the package from npm on next startup. If you prefer to install it yourself:

```bash
cd ~/.config/opencode
bun add opencode-deepseek-thinking-fix
# or: npm i opencode-deepseek-thinking-fix
# or: pnpm add opencode-deepseek-thinking-fix
```

That's it. Any provider whose id contains `deepseek` will have its fetch wrapped automatically.

### Local development

```json
{
  "plugin": ["file:///absolute/path/to/opencode-deepseek-thinking-fix/dist/index.js"]
}
```

### Verify it's loaded

Start opencode, send a message that triggers thinking, and set `debugThinking: true` on the provider to see log lines:

```json
{
  "provider": {
    "deepseek": {
      "options": { "debugThinking": true }
    }
  }
}
```

You should see entries like `[deepseek-thinking-fix][anthropic] cached N thinking blocks (stream)` during the first turn and `re-injected thinking blocks` on subsequent turns.

---

## What it actually does

1. Hooks `config` to wrap `provider.options.fetch` on every matching provider.
2. On **request**: inspects the JSON body, identifies the protocol, and re-injects cached thinking content onto any assistant history message that is missing it. For Anthropic-shaped requests it also makes sure the `thinking` parameter stays enabled so the server keeps producing blocks.
3. On **response**: parses the body (streamed or non-streamed) and caches the thinking content against a stable fingerprint of the message prefix.
4. If anything unexpected happens, the original request is passed through untouched — the plugin never breaks a chat.

### When does it trigger?

Only when **all** of the following hold; otherwise the request is a no-op pass-through:

- Provider id contains one of: `deepseek`, `deep-seek`, `ds-v4`, `dsv4` (case-insensitive).
- `init.body` is a JSON string with a `messages` array.
- URL or body shape looks like Anthropic Messages **or** OpenAI Chat Completions.
- Response is `2xx`.

### Protocol detection

1. URL contains `/v1/messages` or ends with `/messages` → Anthropic.
2. URL contains `/chat/completions` → OpenAI.
3. Fallback by body shape:
   - top-level `system` field or `thinking` param → Anthropic
   - `system` / `developer` role inside `messages`, or `reasoning_effort` at the top → OpenAI
4. Otherwise: pass through unchanged.

---

## Programmatic use

Use the lower-level helper directly if you are embedding opencode or building your own fetch chain:

```ts
import { wrapFetchForDeepSeekThinking } from "opencode-deepseek-thinking-fix";

const fetchWithFix = wrapFetchForDeepSeekThinking(globalThis.fetch, {
  debug: true,
  ttlMs: 30 * 60 * 1000,
  ensureThinkingEnabled: true,
  defaultBudgetTokens: 8000,
  handleOpenAI: true,
});
```

### Options

| Option | Default | Description |
|---|---|---|
| `ttlMs` | `1800000` (30 min) | TTL of the thinking-content cache |
| `debug` | `false` | Extra `console.log` lines for injection / cache hits |
| `ensureThinkingEnabled` | `true` | Auto-add `thinking: { type: "enabled", budget_tokens: N }` to Anthropic requests |
| `defaultBudgetTokens` | `8000` | Budget tokens used by the above |
| `handleOpenAI` | `true` | Also handle OpenAI-compatible `/chat/completions` bodies |

---

## How the fingerprint works

Thinking content is cached under an FNV-1a hash of the message prefix that preceded the assistant turn. The hash **excludes** previously-cached thinking blocks, signatures, and text whitespace, and **includes** `model`, `role`, text content, tool_use / tool_result payloads, and the protocol tag. That way the Anthropic and OpenAI caches never collide, and a retry with the same history always hits the same slot.

---

## FAQ

**Does it work with newapi / OpenRouter / one-api / LiteLLM?**
Yes, as long as you name the opencode provider with `deepseek` in its id (or any of the fallback tokens) and the relay forwards either Anthropic `/v1/messages` or OpenAI `/chat/completions` traffic.

**Does it touch non-DeepSeek providers?**
No. Providers whose id doesn't match are left completely alone.

**Does it break non-thinking / non-reasoning calls?**
No. If the request doesn't have an assistant history that needs patching, nothing changes. If the response contains no thinking content, nothing is cached.

**Where is the cache?**
In-memory, per opencode process. TTL is 30 minutes by default.

**Will it leak thinking content to the server?**
It only re-sends the thinking blocks that the server itself returned in the previous turn. Nothing new is synthesized.

---

## Build from source

```bash
bun install
bun run build
```

Produces `dist/` with ESM + `.d.ts` + sourcemaps.

---

## Links

- npm: https://www.npmjs.com/package/opencode-deepseek-thinking-fix
- OpenCode plugins docs: https://opencode.ai/docs/plugins/
- Related upstream issue: https://github.com/anomalyco/opencode/issues/16748

---

## License

MIT
