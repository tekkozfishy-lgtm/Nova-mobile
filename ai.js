import { pipeline } from
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

let generator = null;
let loading = false;

/*
    NOVA LOCAL AI

    The model runs in the browser.
    No OpenAI API key is required.

    We start with a small model so we can
    test whether your iPhone can handle it.
*/

const MODEL =
    "onnx-community/Qwen2.5-0.5B-Instruct";

async function loadAI() {

    if (generator) {
        return generator;
    }

    if (loading) {

        while (loading) {
            await new Promise(
                resolve => setTimeout(resolve, 100)
            );
        }

        return generator;
    }

    loading = true;

    try {

        console.log(
            "NOVA: Loading local AI..."
        );

        const useWebGPU =
            "gpu" in navigator;

        generator = await pipeline(
            "text-generation",
            MODEL,
            {
                device:
                    useWebGPU
                        ? "webgpu"
                        : "wasm",

                dtype:
                    useWebGPU
                        ? "q4"
                        : "q4"
            }
        );

        console.log(
            "NOVA: Local AI ready."
        );

        return generator;

    } catch (error) {

        console.error(
            "NOVA AI failed to load:",
            error
        );

        generator = null;

        throw error;

    } finally {

        loading = false;

    }

}


async function askNOVA(
    userMessage
) {

    const ai =
        await loadAI();

    const prompt = `
You are NOVA, a friendly and intelligent
personal assistant.

Be helpful, natural and concise.

User:
${userMessage}

NOVA:
`;

    const result =
        await ai(
            prompt,
            {
                max_new_tokens: 100,

                temperature: 0.7,

                do_sample: true
            }
        );

    return result[0].generated_text
        .split("NOVA:")
        .pop()
        .trim();

}


/*
    Make the functions available
    to our main NOVA application.
*/

window.NOVA_AI = {

    load: loadAI,

    ask: askNOVA

};
