/**
 * Programmatic API. Import from `opencode-deepseek-thinking-fix/api`
 * (not from the package root) so opencode's plugin loader doesn't try to
 * invoke these helpers as plugin entries.
 */
export { ThinkingCache, fingerprint } from "./cache.js";
export type { ThinkingBlock } from "./cache.js";
export { assistantTurnFingerprint } from "./messages.js";
export type { AnthropicRequestBody } from "./messages.js";
export {
  fillReasoningPlaceholder,
  reinjectReasoningContent,
  extractReasoningFromOpenAIResponse,
  assistantTurnFingerprintOpenAI,
} from "./messages-openai.js";
export type {
  OpenAIChatRequestBody,
  OpenAIReasoningPayload,
  PlaceholderOptions,
} from "./messages-openai.js";
