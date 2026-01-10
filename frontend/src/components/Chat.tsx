import {useState, useRef, useEffect} from 'react';
import type {KeyboardEvent} from 'react';
import {Button, Input} from '@heroui/react';
import {FaPaperclip, FaRegTrashAlt} from "react-icons/fa";

type Message = {
    id: string;
    text: string;
    fromUser: boolean;
};

type Props = {
    setPdfFile: (f?: File) => void;
    handleNewDocument: () => void;
}

export default function Chat({setPdfFile, handleNewDocument}:Props) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const listRef = useRef<HTMLDivElement | null>(null);

    const sendMessage = () => {
        const trimmed = input.trim();
        if (!trimmed) return;
        const msg: Message = { id: String(Date.now()), text: trimmed, fromUser: true };
        setMessages((m) => [...m, msg]);
        setInput('');
        // For now echo the message as a placeholder bot response
        setTimeout(() => {
            const bot: Message = { id: String(Date.now() + 1), text: `Echo: ${trimmed}`, fromUser: false };
            setMessages((m) => [...m, bot]);
        }, 300);
    };

    const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
    };

    // Scroll to bottom whenever messages change
    useEffect(() => {
        const el = listRef.current;
        if (!el) return;
        // give the browser a tick to render the new message
        requestAnimationFrame(() => {
            el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
        });
    }, [messages]);

    return (
        <div className="p-4 bg-white rounded-xl shadow flex flex-col gap-4 h-full min-h-[420px] ">
            <h3 className="text-lg font-semibold">Chat</h3>
            <div ref={listRef} className="flex-1 overflow-auto flex flex-col gap-2 p-4 bg-gray-50 rounded">
                {messages.length === 0 && (
                    <div className="text-sm text-gray-500">No messages yet. Ask something about the document.</div>
                )}
                {messages.map((m) => (
                    <div key={m.id} className={`px-3 py-2 rounded-xl max-w-[80%] ${m.fromUser ? 'self-end bg-primary-100 text-primary-900' : 'self-start bg-gray-200 text-gray-900'}`}>
                        {m.text}
                    </div>
                ))}
            </div>

            <div className="flex gap-2">
                <Button variant={"light"} color={"danger"} onPress={() => setPdfFile(undefined)} isIconOnly={true}>
                    <FaRegTrashAlt />
                </Button>
                <Button color={"primary"} onPress={handleNewDocument} isIconOnly={true} variant={"light"}>
                    <FaPaperclip />
                </Button>
                <Input
                    placeholder="Type a message"
                    value={input}
                    onChange={(e) => setInput((e.target as HTMLInputElement).value)}
                    onKeyDown={onKeyDown}
                    className="flex-1 min-w-48"
                />
                <Button color="primary" onPress={sendMessage}>Send</Button>
            </div>
        </div>
    );
}
