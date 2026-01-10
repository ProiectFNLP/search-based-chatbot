import {useRef, useState} from "react";
import {pdfjs} from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import {addToast, Tab, Tabs} from "@heroui/react";
import {SelectDocument, PdfViewer, Chat} from "./components";
import SearchPanel from "./components/SearchPanel";
import {assert_error} from "./assertions.ts";
import type {DragEvent} from "react";
import {uploadEndpoint} from "./utils/api.ts";
import {IoChatboxOutline} from "react-icons/io5";
import {FaSearch} from "react-icons/fa";

// Set worker source for pdf.js
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/build/pdf.worker.min.mjs',
    import.meta.url,
).toString();


function App() {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [pdfFile, setPdfFile] = useState<File>();
    const [dragOver, setDragOver] = useState(false);
    const [numPages, setNumPages] = useState<number>();
    const [currentPage, setCurrentPage] = useState(1);
    const [session_id, setSessionId] = useState<string>();
    const [uploading, setUploading] = useState(false);

    const handleNewDocument = () => fileInputRef.current?.click();

    const handleFiles = async (files: FileList | null) => {
        console.log("files", files);
        if (!files) return;
        const file = files[0];
        if (!file || file.type !== "application/pdf") {
            addToast({
                title: "Only PDF files are supported.",
                severity: "danger",
            });
            return;
        }
        setUploading(true);
        setPdfFile(file);
        setCurrentPage(1);
        setSessionId(undefined);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const {session_id} = await uploadEndpoint(formData, {});
            setSessionId(session_id);
        } catch (error: unknown) {
            if(!assert_error(error)) return;
            console.error("Error uploading PDF:", error);
            addToast({
                title: "Error uploading PDF",
                severity: "danger",
                description: error.message,
            });
        } finally {
            setUploading(false);
        }
    };

    const handleDrop = async (e: DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        await handleFiles(e.dataTransfer.files);
    };

    return (
        <div className="h-screen max-h-screen bg-gray-50 p-5 flex overflow-y-scroll">
            <div className={"flex flex-col items-center justify-center w-full h-full "}>
                {!(pdfFile && session_id) &&
                    <SelectDocument
                        handleDrop={handleDrop}
                        handleClick={handleNewDocument}
                        setDragOver={setDragOver}
                        dragOver={dragOver}
                        loading={uploading}
                    />
                }
                <input
                    type="file"
                    accept="application/pdf"
                    multiple={false}
                    ref={fileInputRef}
                    className="hidden"
                    onChange={(e) => handleFiles(e.target.files)}
                />

                {pdfFile && session_id && (
                    <div className={"flex flex-row gap-10 h-min max-h-full w-11/12 justify-center p-10"}>
                        <PdfViewer
                            key={pdfFile.name + pdfFile.size}
                            file={pdfFile}
                            currentPage={currentPage}
                            setCurrentPage={setCurrentPage}
                            numPages={numPages}
                            setNumPages={setNumPages}
                        />
                        <div className="flex w-full flex-col h-full">
                            <Tabs aria-label="Options"
                                  classNames={{
                                      tabList: "gap-6 w-full relative rounded-none p-0 border-b border-divider",
                                      cursor: "w-full bg-primary-600",
                                      tab: "max-w-fit px-0 h-12",
                                      tabContent: "group-data-[selected=true]:text-primary-600",
                                  }}
                                  color="primary"
                                  variant="underlined"
                            >
                                <Tab key={"chat"}
                                     title={
                                         <div className="flex items-center space-x-2">
                                             <IoChatboxOutline />
                                             <span>Chat</span>
                                         </div>
                                     }
                                     className={"h-full"}
                                >
                                    <Chat setPdfFile={setPdfFile} handleNewDocument={handleNewDocument} session_id={session_id} />
                                </Tab>
                                <Tab key={"search"}
                                     title={
                                         <div className="flex items-center space-x-2">
                                             <FaSearch />
                                             <span>Search</span>
                                         </div>
                                     }
                                     className={"h-full"}
                                >
                                    <SearchPanel
                                        pdfFile={pdfFile}
                                        session_id={session_id}
                                        setCurrentPage={setCurrentPage}
                                        handleNewDocument={handleNewDocument}
                                        setPdfFile={setPdfFile}
                                    />
                                </Tab>
                            </Tabs>
                        </div>

                    </div>
                )}

            </div>

        </div>
    )
        ;
}

export default App;

