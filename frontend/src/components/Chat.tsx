import {useState, useRef, useEffect, type KeyboardEvent} from 'react';
import {addToast, Button, Input} from '@heroui/react';
import {FaPaperclip, FaRegTrashAlt} from "react-icons/fa";
import {assert_error} from "../assertions.ts";
import {sendMessageEndpoint} from "../utils/api.ts";
import {MessageBubble} from "./MessageBubble.tsx";
import type {Message} from "../types/messageTypes.ts";

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
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [messageCounter, setMessageCounter] = useState(0);
    const listRef = useRef<HTMLDivElement | null>(null);
    const eventSourceRef = useRef<EventSource | null>(null);

    const handleSendMessage = async () => {
        if (!session_id) return;
        const trimmedMessage = input.trim();
        if (!trimmedMessage) return;
        let msgId = messageCounter;
        const msg: Message = {id: String(msgId++), text: trimmedMessage, fromUser: true, streamingChunk: "", isStreaming: false};
        setMessages((m) => [...m, msg]);
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
            const eventSource: EventSource = new EventSource(sendMessageEndpoint({
                session_id,
                message: trimmedMessage
            }));
            setMessages((m) => [...m, {id: String(msgId++), text: '', fromUser: false, streamingChunk: "", isStreaming: true}]);
            setMessageCounter(msgId);
            eventSourceRef.current = eventSource;

            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log("SSE data:", data);
                setMessages(prev => {
                    const last = prev[prev.length - 1];

                    if (!last || last.fromUser) return prev;

                    return [
                        ...prev.slice(0, -1),
                        {
                            ...last,
                            text: last.text + last.streamingChunk,
                            isStreaming: true,
                            streamingChunk: data.response,
                        }
                    ];
                });

            };

            eventSource.onerror = (err) => {
                console.error("SSE error:", err);
                try {
                    eventSource.close();
                } catch (e) { /* ignore */
                }
                if (eventSourceRef.current === eventSource) eventSourceRef.current = null;
                setMessages(prev => {
                    const last = prev[prev.length - 1];

                    return [
                        ...prev.slice(0, -1),
                        { ...last, isStreaming: false, streamingChunk: "", text: last.text + last.streamingChunk },
                    ];
                });
                setLoading(false);
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
            handleSendMessage();
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
            <h3 className="text-lg font-semibold text-center p-2 flex justify-center"><span>Chat</span></h3>
            <div ref={listRef} className="flex-1 overflow-auto flex flex-col gap-2 p-4 bg-gray-50 border-y-1">
                {messages.length === 0 && (
                    <div className="text-sm text-gray-500">No messages yet. Ask something about the document.</div>
                )}
                {messages.map((m) => (
                    <MessageBubble key={m.id} message={m}/>
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
