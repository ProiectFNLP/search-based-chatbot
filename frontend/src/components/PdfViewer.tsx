import { Document, Page } from "react-pdf";
import { Divider } from "@heroui/react";
import { PaginationButtons } from "./PaginationButtons";

type PdfViewerProps = {
    file: File;
    currentPage: number;
    setCurrentPage: (page: number) => void;
    numPages?: number;
    setNumPages: (numPages: number) => void;
};

export function PdfViewer({
                              file,
                              currentPage,
                              setCurrentPage,
                              numPages,
                              setNumPages,
                          }: PdfViewerProps) {
    return (
        <div className="w-1/2 max-w-3xl min-w-fit h-full bg-white p-4 rounded-xl shadow-lg flex flex-col items-center">
            <Document
                file={file}
                onLoadSuccess={({ numPages }) => setNumPages(numPages)}
                className={"shrink grow min-h-0"}
            >
                <Page
                    pageNumber={currentPage}
                    renderAnnotationLayer={false}
                    renderTextLayer={false}
                    scale={0.92}
                    // className={'flex flex-col items-center'}
                >
                </Page>
            </Document>

            <Divider />

            {numPages && (
                <PaginationButtons
                    currentPage={currentPage}
                    setCurrentPage={setCurrentPage}
                    numPages={numPages}
                />
            )}
        </div>
    );
}
