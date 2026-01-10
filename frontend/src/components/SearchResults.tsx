type SearchResultItem = {
    page: number;
    score: number;
    first: boolean;
};

type SearchResultsProps = {
    results: SearchResultItem[];
    onSelectPage: (page: number) => void;
};

export function SearchResults({ results, onSelectPage }: SearchResultsProps) {
    if (!results.length) return null;

    return (
        <ul className="divide-y">
            {results.map((item) => (
                <li key={item.page}>
                    <button
                        type="button"
                        onClick={() => onSelectPage(item.page)}
                        className={`w-full text-left px-4 py-2 hover:bg-gray-100 ${item.first ? 'bg-primary-50' : ''}`}
                    >
                        <div className="flex flex-col">
                            <span className="font-medium">Page {item.page}</span>
                            <span className="text-xs text-gray-500">Score: {item.score}</span>
                        </div>
                    </button>
                </li>
            ))}
        </ul>
    );
}
