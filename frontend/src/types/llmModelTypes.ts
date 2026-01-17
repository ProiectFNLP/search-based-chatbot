const LlmModelTypeValues = ["gpt-5-nano" , "llama", "flan-t5-base", "qwen2-1.5b"] as const;
type LlmModelType = typeof LlmModelTypeValues[number];

export {LlmModelTypeValues};
export type { LlmModelType };