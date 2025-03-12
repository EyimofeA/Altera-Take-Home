## What should you do if the two models have different tokenizers?

If the two models use different tokenizers, you need to ensure that the input text is processed in a compatible way for both models. Here are some ways to do this:

- **Adopt a Common Tokenizer:**  
  Use one tokenizer (typically the one from the more capable or larger model) to preprocess the input, and then adjust or re-tokenize outputs as necessary for the other model.

- **Create a Mapping Between Tokenizers:**  
  Develop a conversion layer that maps tokens from one vocabulary to the other.

- **Retrain or Fine-Tune Tokenizers:**  
  Aligning the tokenizers by fine-tuning or retraining one model’s tokenizer to match the vocabulary of the other model might be necessary to ensure consistency in token representation.

## Do you think contrastive decoding is used in practice?

Yes, but primarily in research settings.

Contrastive decoding can improve output quality by leveraging the strengths of two models, making it a useful technique when quality is a priority. However, it incurs significantly higher computational costs than traditional decoding methods like beam search, top-k sampling, or nucleus sampling.

Due to this tradeoff, it is more likely to be used in research and experimentation rather than large-scale production systems, where efficiency and latency are crucial. That said, for specialized applications where quality outweighs computational cost, it may find practical use.
