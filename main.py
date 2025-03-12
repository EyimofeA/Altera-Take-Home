import transformers as tr
import torch
from contextlib import nullcontext

# Define model paths.
amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-3B-Instruct"

# Load tokenizer from the smaller model.
tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)

EPS = 1e-10  # Constant for numerical stability.

def contrastive_generation(amateur: tr.PreTrainedModel, expert: tr.PreTrainedModel, prompt: str, max_tokens: int, 
                           alpha: float = 0.1, temperature: float = 1.0, ablation: bool = False) -> str:
    """
    Implements contrastive decoding (https://arxiv.org/abs/2210.15097) at token-level granularity
    with caching for efficiency.

    Args:
        amateur: The smaller model to penalize undesirable token behaviors.
        expert: The larger model providing high-quality logits.
        prompt (str): Input text to initiate generation.
        max_tokens (int): Maximum number of tokens to generate.
        alpha (float): Threshold hyperparameter for plausible token selection.
        temperature (float): Temperature scaling factor for the amateur model's logits.
        ablation (bool): When True, processes only the last token for the amateur model without full context.

    Returns:
        str: The generated continuation text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(amateur.device)
    input_ids = inputs["input_ids"]
    generated_tokens = input_ids.clone()

    amateur.eval()
    expert.eval()

    # Initialize caches with the full prompt.
    with torch.no_grad():
        amateur_out = amateur(generated_tokens, use_cache=True)
        expert_out = expert(generated_tokens, use_cache=True)
        past_amateur = amateur_out.past_key_values
        past_expert = expert_out.past_key_values

    # Use mixed precision when on CUDA.
    device = generated_tokens.device
    autocast_context = torch.amp.autocast('cuda') if device.type == "cuda" else nullcontext()

    for _ in range(max_tokens):
        with torch.no_grad(), autocast_context:
            # Process only the last generated token.
            last_token = generated_tokens[:, -1:]
            
            # Ablation mode: process the last token without past context.
            if ablation:
                amateur_out = amateur(last_token, use_cache=True)
            else:
                # Normal mode: use cached past to process only the new token.
                amateur_out = amateur(last_token, past_key_values=past_amateur, use_cache=True)
            
            # Expert model always uses cached context.
            expert_out = expert(last_token, past_key_values=past_expert, use_cache=True)
            
            # Update caches.
            past_amateur = amateur_out.past_key_values
            past_expert = expert_out.past_key_values

            # Apply temperature scaling on amateur logits.
            amateur_logits = amateur_out.logits[:, -1, :] / temperature
            expert_logits = expert_out.logits[:, -1, :]

            # Compute log-softmax for both models.
            expert_log_probs = torch.log_softmax(expert_logits, dim=-1)
            amateur_log_probs = torch.log_softmax(amateur_logits, dim=-1)
            
            # Recover expert probabilities.
            expert_probs = torch.exp(expert_log_probs)

            # Create a plausibility mask.
            max_expert_prob = torch.max(expert_probs)
            plausible_mask = expert_probs >= (alpha * max_expert_prob)

            # Compute contrastive scores and mask out implausible tokens.
            contrastive_scores = expert_log_probs - amateur_log_probs
            contrastive_scores.masked_fill_(~plausible_mask, float("-inf"))

            # Select the token with the highest contrastive score.
            next_token = torch.argmax(contrastive_scores, dim=-1, keepdim=True)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            # Stop if the EOS token is generated.
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated tokens (excluding the prompt).
    return tokenizer.decode(generated_tokens[0][input_ids.shape[1]:], skip_special_tokens=True)

def main():
	user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
		scores,
		results,
		kFactor = 4,
	) {
		for (const result of results) {
			const { first, second, outcome } = result;
			const firstScore = scores[first] ?? 1000;
			const secondScore = scores[second] ?? 1000;

			const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
			const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
			let sa = 0.5;
			if (outcome === 1) {
				sa = 1;
			} else if (outcome === -1) {
				sa = 0;
			}
			scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
			scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
		}
		return scores;
	}\n```"""

	prompt = tokenizer.apply_chat_template(
		[
			{"role": "system", "content": "You are a helpful assistant"},
			{"role": "user", "content": user_message},
		],
		add_generation_prompt=True,
		tokenize=False,
	)

	# Select device and enable benchmark mode for CUDA.
	device = "cuda" if torch.cuda.is_available() else "cpu"
	if device == "cuda":
		torch.backends.cudnn.benchmark = True

	# Load models and move them to the selected device.
	amateur_model = tr.AutoModelForCausalLM.from_pretrained(amateur_path).to(device)
	expert_model = tr.AutoModelForCausalLM.from_pretrained(expert_path).to(device)

	# Convert models to half precision for faster inference if on CUDA.
	if device == "cuda":
		amateur_model.half()
		expert_model.half()
		
	# Compile models using torch.compile for further performance improvements (requires PyTorch 2.0+)
	use_compile = False #torch compile might not work out of the box so its disabled by default
	if use_compile and hasattr(torch, "compile"):
		amateur_model = torch.compile(amateur_model)
		expert_model = torch.compile(expert_model)

	print("Generating output...\n")
	output = contrastive_generation(amateur_model, expert_model, prompt, max_tokens=1000)
	print("Generated Text:\n", output)

if __name__ == "__main__":
    main()
