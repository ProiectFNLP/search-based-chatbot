import {FaSearch} from "react-icons/fa";
import {Button, Input, Progress, Select, SelectItem, addToast} from "@heroui/react";
import { SearchResults } from "./SearchResults";
import {SearchTypeValues} from "../types/searchTypes";
import type {SearchType} from "../types/searchTypes";
import {useEffect, useMemo, useRef, useState} from "react";
import {searchBM25Endpoint, searchFaissEndpoint, searchTfIdfEndpoint} from "../utils/api";
import {assert_error} from "../assertions";
import type {SearchResultsType} from "../types/searchTypes";


type Props = {
    // minimal props: file/session from App and a few handlers that must remain in App
    pdfFile?: File;
    session_id?: string;
    setCurrentPage: (p: number) => void;
    handleNewDocument: () => void; // still triggers file input in App
    setPdfFile: (f?: File) => void;
};

export default function SearchPanel({
     pdfFile,
     session_id,
     setCurrentPage,
     handleNewDocument,
     setPdfFile,
 }: Props) {
     const [searchMode, setSearchMode] = useState<SearchType>("TF-IDF");
     const [searchInput, setSearchInput] = useState("");
     const [loading, setLoading] = useState(false);
     const [searchResults, setSearchResults] = useState<SearchResultsType | null>(null);
     const [numResults] = useState(5);
     const eventSourceRef = useRef<EventSource | null>(null);

     const progress = useMemo(() => {
         if (!loading) return null;
         if (!searchResults) return 0;
         return (searchResults.current + 1) / searchResults.total * 100;
     }, [loading, searchResults]);

     const filteredSearchResults = useMemo(() => {
         if (!searchResults) return null;
         return searchResults.results
             .slice(0, numResults)
             .filter((result) => {
                 return result.score > 0;
             }).map((result, index) => ({
                 page: result.page,
                 score: result.score,
                 first: index === 0,
             }));

     }, [searchResults, numResults]);

     const handleSearch = async () => {
         if(!session_id) return;
         const trimmedSearch = searchInput.trim();
         if (!pdfFile || !trimmedSearch) return;

         // Close any existing EventSource before starting a new one
         if (eventSourceRef.current) {
             try { eventSourceRef.current.close(); } catch (e) { /* ignore */ }
             eventSourceRef.current = null;
         }

         setLoading(true);
         setSearchResults(null);

         try {
             let eventSource: EventSource;
             if (searchMode === "TF-IDF")
                 eventSource = new EventSource(searchTfIdfEndpoint({session_id, search: trimmedSearch}));
             else if (searchMode === "FAISS")
                 eventSource = new EventSource(searchFaissEndpoint({session_id, search: trimmedSearch}));
             else if (searchMode === "BM-25")
                 eventSource = new EventSource(searchBM25Endpoint({session_id, search: trimmedSearch}));
             else
                 throw new Error("Unknown search type");

             eventSourceRef.current = eventSource;

             eventSource.onmessage = (event) => {
                 const data = JSON.parse(event.data);
                 console.log("SSE data:", data);
                 setSearchResults(data);
             };

             eventSource.onerror = (err) => {
                 console.error("SSE error:", err);
                 try { eventSource.close(); } catch (e) { /* ignore */ }
                 if (eventSourceRef.current === eventSource) eventSourceRef.current = null;
                 setLoading(false);
             };

             eventSource.onopen = () => console.log("Connected to SSE");
         } catch (error: unknown) {
             if(!assert_error(error)) return;
             console.error("Error sending PDF:", error);
             addToast({
                 title: "Error sending PDF",
                 severity: "danger",
                 description: (error as Error).message,
             });
             throw error;

         }
     }

     useEffect(() => {
         if (!loading && searchResults) {
             setCurrentPage(searchResults.results[0].page)
         }
     }, [loading, searchResults, setCurrentPage]);

     useEffect(() => {
         return () => {
             if (eventSourceRef.current) {
                 try { eventSourceRef.current.close(); } catch (e) { /* ignore */ }
                 eventSourceRef.current = null;
             }
         };
     }, []);

     return (
         <div className={"p-4 flex flex-col items-start gap-4 justify-between h-full w-1/2" }>
             <div className={"flex flex-col w-full gap-2"}>
                 <p>Search mode</p>
                 <Select
                     selectedKeys={new Set([searchMode])}
                     onSelectionChange={(e) => e.currentKey && setSearchMode(e.currentKey as SearchType)}
                     aria-label="Search Mode"
                     labelPlacement={"outside"}
                     color={"primary"}
                 >
                     {
                         SearchTypeValues.map((item) => (
                             <SelectItem key={item}>{item}</SelectItem>
                         ))
                     }
                 </Select>
                 <p>Search in document</p>
                 <div className={"flex flex-row gap-4 w-full"}>
                     <Input
                         startContent={<FaSearch/>}
                         placeholder="Search"
                         color={"primary"}
                         type="search"
                         value={searchInput}
                         onChange={(e) => setSearchInput((e.target as HTMLInputElement).value)}
                         className="w-full"
                     />
                     <Button
                         onPress={handleSearch}
                         variant={"bordered"}
                         isLoading={loading}
                         color={"primary"}>Search</Button>
                 </div>
                 {progress !== null &&
                     <Progress aria-label="Loading..." className="max-w-md" value={progress}/>}
             </div>
             <div className={"flex flex-col w-full gap-6 self-center"}>
                 {filteredSearchResults && (
                    <SearchResults
                        results={filteredSearchResults}
                        onSelectPage={setCurrentPage}
                    />
                 )}

             </div>
             <div className={"flex flex-row gap-4"}>
                 <Button
                     color={"primary"}
                     onPress={() => handleNewDocument()}>
                     New Document
                 </Button>
                 <Button
                     variant={"bordered"}
                     color={"danger"}
                     onPress={() => setPdfFile(undefined)}
                 >
                     Remove Document
                 </Button>
             </div>
         </div>
     );
 }
