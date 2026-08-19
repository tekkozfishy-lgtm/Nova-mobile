import { pipeline } from
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

let novaAI = null;
let loadingPromise = null;

const MODEL =
    "onnx-community/SmolLM2-135M-Instruct-ONNX-MHA";


async function loadAI() {

    if (novaAI) {
        return novaAI;
    }

    if (loadingPromise) {
        return loadingPromise;
    }

    loadingPromise = (async () => {

        console.log(
            "NOVA: Loading local AI..."
        );

        const options = {
            dtype: "q4"
        };

        /*
         * Use WebGPU when available.
         * Otherwise Transformers.js
         * falls back to WASM.
         */

        if ("gpu" in navigator) {

            options.device = "webgpu";

        } else {

            options.device = "wasm";

        }

        novaAI = await pipeline(
            "text-generation",
            MODEL,
            options
        );

        console.log(
            "NOVA: Local AI ready."
        );

        return novaAI;

    })();

    try {

        return await loadingPromise;

    } catch (error) {

        console.error(
            "NOVA: Failed to load AI:",
            error
        );

        novaAI = null;
        loadingPromise = null;

        throw error;

    }

}


async function askNOVA(
    message
) {

    const ai =
        await loadAI();


    const messages = [

        {
            role: "system",

            content:
                "You are NOVA, a friendly " +
                "personal assistant. " +
                "Give concise, natural answers."
        },

        {
            role: "user",

            content: message
        }

    ];


    const result =
        await ai(
            messages,
            {
                max_new_tokens: 80,

                temperature: 0.7,

                do_sample: true
            }
        );


    const generated =
        result[0].generated_text;


    /*
     * Transformers.js may return the
     * complete conversation, so extract
     * the assistant's response.
     */

    if (
        Array.isArray(
            generated
        )
    ) {

        const last =
            generated[
                generated.length - 1
            ];

        if (
            last &&
            last.content
        ) {

            return last.content.trim();

        }

    }


    if (
        typeof generated ===
        "string"
    ) {

        return generated.trim();

    }


    return (
        "I generated a response, " +
        "but couldn't read it."
    );

}


window.NOVA_AI = {

    load: loadAI,

    ask: askNOVA

};

console.log(
    "NOVA local AI module loaded."
);
