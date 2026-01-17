import {useState, useRef, useEffect, type KeyboardEvent} from 'react';
import {addToast, Button, Input, Select, SelectItem} from '@heroui/react';
import {FaPaperclip, FaRegTrashAlt} from "react-icons/fa";
import {assert_error} from "../assertions.ts";
import {sendPromptEndpoint} from "../utils/api.ts";
import {MessageBubble} from "./MessageBubble.tsx";
import {useChat} from '../contexts/ChatContext';
import {type SearchType, SearchTypeValues} from "../types/searchTypes";
import {type LlmModelType, LlmModelTypeValues} from "../types/llmModelTypes";

type Props = {
    setPdfFile: (f?: File) => void;
    handleNewDocument: () => void;
    session_id?: string;
}

export default function Chat({
                                 setPdfFile,
                                 handleNewDocument,
                                 session_id
                             }: Props) {
    const {messages, addUserMessage, addAssistantPlaceholder, appendStreamingChunk, finalizeMessage} = useChat();
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [searchMode, setSearchMode] = useState<SearchType>("TF-IDF");
    const [llmModel, setLlmModel] = useState<LlmModelType>("gpt-5-nano");
    const listRef = useRef<HTMLDivElement | null>(null);
    const eventSourceRef = useRef<EventSource | null>(null);

    const handleSendMessage = async () => {
        if (!session_id) return;
        const trimmedMessage = input.trim();
        if (!trimmedMessage) return;

        // add user message to global context
        addUserMessage(trimmedMessage);
        setInput('');

        if (eventSourceRef.current) {
            try {
                eventSourceRef.current.close();
            } catch (e) { /* ignore */
            }
            eventSourceRef.current = null;
        }

        setLoading(true);

        try {
            const eventSource: EventSource = new EventSource(sendPromptEndpoint({
                session_id,
                prompt: trimmedMessage,
                search_mode: searchMode.toLowerCase().replace("-", ""),
                llm_model: llmModel,
            }));

            // add assistant placeholder
            const assistantId = addAssistantPlaceholder();

            eventSourceRef.current = eventSource;

            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log("SSE data:", data);
                console.log(data?.done)
                if(data?.done){
                    try {
                        eventSource.close();
                    } catch (e) { /* ignore */
                    }
                    if (eventSourceRef.current === eventSource) eventSourceRef.current = null;
                    finalizeMessage(assistantId);
                    setLoading(false);
                } else {
                    appendStreamingChunk(assistantId, data.response || "");
                }
            };

            eventSource.onerror = (err) => {
                console.error("SSE error:", err);
                try {
                    eventSource.close();
                } catch (e) { /* ignore */
                }
                if (eventSourceRef.current === eventSource) eventSourceRef.current = null;
            };

            eventSource.onopen = () => console.log("Connected to SSE");
        } catch (error: unknown) {
            if (!assert_error(error)) return;
            console.error("Error sending the message:", error);
            addToast({
                title: "Error sending the message",
                severity: "danger",
                description: (error as Error).message,
            });
            throw error;

        }
    }


    const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            void handleSendMessage();
        }
    };

    useEffect(() => {
        console.log(messages.map(m => m.id));
        const el = listRef.current;
        if (!el) return;

        requestAnimationFrame(() => {
            el.scrollTo({top: el.scrollHeight, behavior: 'smooth'});
        });
    }, [messages]);

    return (
        <div className="bg-white rounded-xl shadow-lg flex flex-col max-h-s h-full min-h-[420px] ">
            <h3 className="relative text-lg font-semibold p-2 min-w-[500px]">
                <span className="absolute left-1/2 -translate-x-1/2">
                    Chat
                </span>

                <div className="flex flex-row gap-2 ml-auto justify-end w-1/3 min-w-[200px]">
                    <Select
                        selectedKeys={new Set([searchMode])}
                        onSelectionChange={(e) => e.currentKey && setSearchMode(e.currentKey as SearchType)}
                        aria-label="Search Mode"
                        labelPlacement="outside"
                        color="primary"
                        className={"w-xs"}
                    >
                        {SearchTypeValues.map((item) => (
                            <SelectItem key={item}>{item}</SelectItem>
                        ))}
                    </Select>

                    <Select
                        selectedKeys={new Set([llmModel])}
                        onSelectionChange={(e) => e.currentKey && setLlmModel(e.currentKey as LlmModelType)}
                        aria-label="LLM Model"
                        labelPlacement="outside"
                        color="primary"
                    >
                        {LlmModelTypeValues.map((item) => (
                            <SelectItem key={item}>{item}</SelectItem>
                        ))}
                    </Select>
                </div>
            </h3>

            <div ref={listRef} className="flex-1 overflow-auto flex flex-col gap-2 p-4 bg-gray-50 border-y-1">
                {messages.length === 0 && (
                    <div className="text-sm text-gray-500">No messages yet. Ask something about the document.</div>
                )}
                {messages.map((m, index) => (
                    <MessageBubble key={index} message={m}/>
                ))}
            </div>

            <div className="flex flex-row gap-2 items-center justify-center py-3 px-2">
                <Button variant={"light"} color={"danger"} onPress={() => setPdfFile(undefined)} isIconOnly={true}>
                    <FaRegTrashAlt/>
                </Button>
                <Button color={"primary"} onPress={handleNewDocument} isIconOnly={true} variant={"light"}>
                    <FaPaperclip/>
                </Button>
                <Input
                    placeholder="Type a message"
                    value={input}
                    onChange={(e) => setInput((e.target as HTMLInputElement).value)}
                    onKeyDown={onKeyDown}
                    className="flex-1 min-w-48"
                />
                <Button color="primary" onPress={handleSendMessage} isLoading={loading}>Send</Button>
            </div>
        </div>
    );
}
