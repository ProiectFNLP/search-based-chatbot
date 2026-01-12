const LlmModelTypeValues = ["gpt-4o-mini" , "llama"] as const;
type LlmModelType = typeof LlmModelTypeValues[number];

export {LlmModelTypeValues};
export type { LlmModelType };