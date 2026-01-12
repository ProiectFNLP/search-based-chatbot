// http://localhost:8000/search-tf-idf?session_id=${session_id}&search=${trimmedSearch}

const BASE_URL = 'http://localhost:8000';


function typedEntries<T extends Record<PropertyKey, unknown>>(
    obj: T
): [keyof T, T[keyof T]][] {
    return Object.entries(obj) as any;
}

function makeEndpoint<P extends readonly string[]>(endpoint: string): (args: Record<P[number], string>) => URL {
    return (args) => {
        const url = new URL(endpoint, new URL(BASE_URL));
        for (const [key, value] of typedEntries(args)) {
            url.searchParams.append(key, value);
        }
        return url;
    }
}

type Method = "GET" | "POST" | "PUT" | "DELETE" | "OPTIONS" | "PATCH";

function makeFetch<UrlArgs extends readonly string[], Payload extends BodyInit | object, Response extends Record<string, unknown>>(method: Method, endpoint: string, json: Payload extends BodyInit ? false : true):
    (payload: Payload, endpoint_args: Record<UrlArgs[number], string>) => Promise<Response> {
    return async (payload, args) => {
        const response = await fetch(
            makeEndpoint<UrlArgs>(endpoint)(args),
            {
                method: method,
                body: json ? JSON.stringify(payload) : (payload as BodyInit),
            }
        );
        return await response.json() as Response;
    }
}

const searchTfIdfEndpoint = makeEndpoint<["session_id", "search"]>("search-tf-idf");
const searchFaissEndpoint = makeEndpoint<["session_id", "search"]>("search-faiss");
const searchBM25Endpoint = makeEndpoint<["session_id", "search"]>("search-bm25");
// type Endpoint = makeFetch<UrlParamKeys, Body : FormData | object, Response: object>
// await endpoint(payload, url_params);
const uploadEndpoint = makeFetch<[], FormData, {session_id: string}>("POST", "upload", false);
const sendMessageDummyEndpoint = makeEndpoint<["session_id", "message"]>("send-message-dummy");
const sendPromptEndpoint = makeEndpoint<["session_id", "prompt", "search_mode"]>("send-prompt");

export {makeEndpoint, searchTfIdfEndpoint, searchFaissEndpoint, searchBM25Endpoint, uploadEndpoint, sendMessageDummyEndpoint, sendPromptEndpoint};