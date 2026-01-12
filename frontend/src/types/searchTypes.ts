type SearchResultsType = {
    current: number,
    total: number,
    results: {
        page: number,
        score: number,
    }[]
}
const SearchTypeValues = ["TF-IDF", "FAISS", "BM-25"] as const;
type SearchType = typeof SearchTypeValues[number];

export {SearchTypeValues};
export type { SearchType, SearchResultsType };
