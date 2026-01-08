import { Listbox, ListboxItem } from "@heroui/react";

type SearchResultItem = {
    page: number;
    score: number;
    first: boolean;
};

type SearchResultsProps = {
    results: SearchResultItem[];
    onSelectPage: (page: number) => void;
};

export function SearchResults({
                                  results,
                                  onSelectPage,
                              }: SearchResultsProps) {
    if (!results.length) return null;

    return (
        <div>
            <p className="text-xl">Top Results:</p>

            <Listbox
                aria-label="Search Results"
                items={results}
                onAction={(key) =>
                    onSelectPage(typeof key === "string" ? parseInt(key) : key)
                }
            >
                {(item) => (
                    <ListboxItem
                        key={item.page}
                        color={item.first ? "primary" : "default"}
                    >
                        <div>
                            <p>Page {item.page}</p>
                            <p className="text-xs">Score: {item.score}</p>
                        </div>
                    </ListboxItem>
                )}
            </Listbox>
        </div>
    );
}
