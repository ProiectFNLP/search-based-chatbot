import {createContext, useContext, useReducer} from 'react';
import type {ReactNode} from 'react';
import type {Message} from '../types/messageTypes';

type State = {
    messages: Message[]
}

type Action =
    | {type: 'ADD_MESSAGE'; payload: Message}
    | {type: 'APPEND_CHUNK'; payload: {id: string; chunk: string}}
    | {type: 'FINALIZE'; payload: {id: string}}
    | {type: 'REMOVE'; payload: {id: string}}
    | {type: 'CLEAR'};

const initialState: State = {messages: []};

function reducer(state: State, action: Action): State {
    switch (action.type) {
        case 'ADD_MESSAGE':
            return {messages: [...state.messages, action.payload]};
        case 'APPEND_CHUNK':
            return {
                messages: state.messages.map(m => {
                    if (m.id !== action.payload.id) return m;
                    return {
                        ...m,
                        // append streaming chunk to the visible text progressively
                        text: (m.text || '') + (action.payload.chunk || ''),
                        streamingChunk: action.payload.chunk,
                        isStreaming: true,
                    };
                })
            };
        case 'FINALIZE':
            return {
                messages: state.messages.map(m => m.id === action.payload.id ? {...m, isStreaming: false, streamingChunk: ''} : m)
            };
        case 'REMOVE':
            return {messages: state.messages.filter(m => m.id !== action.payload.id)};
        case 'CLEAR':
            return {messages: []};
        default:
            return state;
    }
}

function genId() {
    return `${Date.now()}-${Math.floor(Math.random() * 100000)}`;
}

type ChatContextType = {
    messages: Message[];
    addUserMessage: (text: string) => string;
    addAssistantPlaceholder: () => string;
    appendStreamingChunk: (id: string, chunk: string) => void;
    finalizeMessage: (id: string) => void;
    removeMessage: (id: string) => void;
    clearMessages: () => void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export function ChatProvider({children}: {children: ReactNode}) {
    const [state, dispatch] = useReducer(reducer, initialState);

    const addUserMessage = (text: string) => {
        const id = genId();
        const msg: Message = {id, text, fromUser: true, isStreaming: false, streamingChunk: ''};
        dispatch({type: 'ADD_MESSAGE', payload: msg});
        return id;
    };

    const addAssistantPlaceholder = () => {
        const id = genId();
        const msg: Message = {id, text: '', fromUser: false, isStreaming: true, streamingChunk: ''};
        dispatch({type: 'ADD_MESSAGE', payload: msg});
        return id;
    };

    const appendStreamingChunk = (id: string, chunk: string) => {
        dispatch({type: 'APPEND_CHUNK', payload: {id, chunk}});
    };

    const finalizeMessage = (id: string) => {
        dispatch({type: 'FINALIZE', payload: {id}});
    };

    const removeMessage = (id: string) => dispatch({type: 'REMOVE', payload: {id}});
    const clearMessages = () => dispatch({type: 'CLEAR'});

    const value: ChatContextType = {
        messages: state.messages,
        addUserMessage,
        addAssistantPlaceholder,
        appendStreamingChunk,
        finalizeMessage,
        removeMessage,
        clearMessages,
    };

    return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat() {
    const ctx = useContext(ChatContext);
    if (!ctx) throw new Error('useChat must be used within ChatProvider');
    return ctx;
}
