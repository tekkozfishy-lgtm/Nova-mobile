import { pipeline } from
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const MODEL =
    "onnx-community/SmolLM2-135M-Instruct-ONNX-MHA";

let ai = null;
let loading = null;

async function loadAI() {

    if (ai) {
        return ai;
    }

    if (loading) {
        return loading;
    }

    loading = (async () => {

        try {

            console.log(
                "NOVA: loading local AI..."
            );

            const options = {
                dtype: "q4"
            };

            /*
             * Chrome/Chromebook can use
             * WebGPU when available.
             *
             * Safari/iPhone falls back
             * to WASM.
             */

            if (
                "gpu" in navigator
            ) {

                options.device =
                    "webgpu";

            } else {

                options.device =
                    "wasm";

            }

            ai = await pipeline(
                "text-generation",
                MODEL,
                options
            );

            console.log(
                "NOVA: local AI ready"
            );

            return ai;

        } catch (error) {

            console.error(
                "NOVA AI failed:",
                error
            );

            ai = null;

            throw error;

        }

    })();

    try {

        return await loading;

    } finally {

        loading = null;

    }

}


async function askNOVA(
    message
) {

    const model =
        await loadAI();

    const prompt =
        "You are NOVA, a friendly " +
        "personal assistant. " +
        "Answer the user's question " +
        "clearly and concisely.\n\n" +
        "User: " +
        message +
        "\nNOVA:";


    const result =
        await model(
            prompt,
            {
                max_new_tokens: 80,
                temperature: 0.7,
                do_sample: true
            }
        );


    let answer =
        result[0].generated_text;


    if (
        typeof answer ===
        "string"
    ) {

        if (
            answer.includes("NOVA:")
        ) {

            answer =
                answer
                    .split("NOVA:")
                    .pop();

        }

        return answer.trim();

    }


    return (
        "Sorry, I couldn't understand " +
        "my own response."
    );

}


window.NOVA_AI = {

    load: loadAI,

    ask: askNOVA

};


console.log(
    "NOVA AI module loaded."
);
