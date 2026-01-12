import {useEffect, useState} from "react";

export function useTypewriter(text: string, speed: number, streaming: boolean) {
    const [displayed, setDisplayed] = useState("");
    const [buffer, setBuffer] = useState("");
    const [done, setDone] = useState(false);
    useEffect(() => {
        setBuffer(prev => prev + text);
    }, [text]);

    useEffect(() => {
        console.log(displayed.length);
        if ((!buffer) || displayed.length >= buffer.length) {
            if(!streaming){
                setDone(true);
            }
            return;
        }
        let stopped = false;
        (new Promise(resolve => setTimeout(resolve, speed))).then(() => {
            if (stopped) return;
            setDisplayed(prev => prev + buffer[prev.length]);
        });

        return () => {
            stopped = true;
        }
    }, [displayed, buffer, speed, streaming]);

    return {done, animatedText: displayed}
}
