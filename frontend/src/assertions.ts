function assert_error(x: unknown): x is Error {
    return !!(x as Error);
}

export {assert_error};