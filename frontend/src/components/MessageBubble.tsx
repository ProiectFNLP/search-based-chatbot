import {useTypewriter} from "../hooks";
import type {Message} from "../types/messageTypes.ts";

export const MessageBubble = ({message} :{message: Message}) => {
    const {animatedText, done} = useTypewriter(
        message.isStreaming ? message.streamingChunk: "",
        15,
        message.isStreaming || false
    );

    return (
        <div
            className={`px-3 py-2 rounded-xl max-w-[80%]
                    ${message.fromUser
                ? "self-end bg-primary-100 text-primary-900"
                : "self-start bg-gray-200 text-gray-900"
            }`}
        >
            <span>{done ?  message.text : animatedText}</span>
            {message.isStreaming && <span className="animate-pulse">| </span>}
        </div>
    );
};