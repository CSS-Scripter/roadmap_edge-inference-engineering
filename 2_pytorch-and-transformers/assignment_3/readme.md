# About transformer encoders and decoders

- Encoders enrich sentence information by look both back and ahead at tokens.
- Decoders use 'causal masking' for autoregressive generation of new tokens.


Encoders allow richer context about the sentence than a pure decoder approach, but a decoder is required for generating new tokens. Encoders can run in parallel, as they don't require causal masking. Without causal masking, we can give context to every token in the entire time dimension at the same time.



# About the models

There are three models to be compared:
- T5 small (~60M params) encoder + decoder
- T5 base (~250M params) encoder + decoder
- GPT2 (~124M params) decoder only
- Bert (~109M params) encoder only


When it comes to performance:
- Bert is the fastest for a 200 token buffer, though comparing bert against T5 and GPT2 is a bit like cheating, as Bert is not a decoder at all, and cannot generate text like T5 and GPT2. Bert is of course fastest since it's only an encoder, which can run in parallel, rather than the sequential autoregressive generation passes through the decoder.
- T5 small takes second place. It outperforms GPT2, even though both models perform autoregressive generation, T5 small does so with less parameters than GPT2. This means it has less calculations to do per pass, and will do it's passes faster.
- GPT2 and T5 base perform about equally. Even though T5 base is about double the size of GPT2, the parameters dedicated to decoding is about equal between GPT2 and T5 base. Meaning that the hefty decoder passes run on about the same parameter counts, and the efficient encoder passes are peanuts compared to the decoder's runtime.
