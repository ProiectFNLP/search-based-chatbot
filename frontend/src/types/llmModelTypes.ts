const LlmModelTypeValues = ["gpt-4o-mini" , "llama", "flan-t5-base"] as const;
type LlmModelType = typeof LlmModelTypeValues[number];

export {LlmModelTypeValues};
export type { LlmModelType };