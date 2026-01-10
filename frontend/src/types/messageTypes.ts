export type Message = {
    id: string;
    text: string;
    fromUser: boolean;
    isStreaming?: boolean;
    streamingChunk: string;
};