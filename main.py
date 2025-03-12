import transformers as tr
import torch

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)


def contrastive_generation(amateur, expert, prompt, max_tokens, alpha=0.1,temperature =1.0,ablation = False) -> str:
	"""
	Implements contrastive decoding (https://arxiv.org/abs/2210.15097) at token-level granularity.

	Args:
					amateur: The smaller (amateur) model to penalize undesirable token behaviors.
					expert: The larger model providing high-quality logits.
					prompt (str): Input text to initiate generation.
					max_tokens (int): Maximum number of tokens to generate.

	Returns:
					str: The generated continuation text.
	"""

	inputs = tokenizer(prompt, return_tensors="pt")
	input_ids = inputs["input_ids"]
	generated_tokens = input_ids.to(amateur.device)

	amateur.eval()
	expert.eval()

	for _ in range(max_tokens):
		with torch.no_grad():
			if ablation:
				amateur_logits = amateur(generated_tokens[:, -1].unsqueeze(0)).logits[:, -1, :]
			else:
				amateur_logits = amateur(generated_tokens).logits[:, -1, :]
			expert_logits = expert(generated_tokens).logits[:, -1, :]
			# Temperature scaling
			amateur_logits = amateur_probs/temperature

			amateur_probs = torch.softmax(amateur_logits, dim=-1)
			expert_probs = torch.softmax(expert_logits, dim=-1)
			

			# Apply plausibility constraint (V_head)
			max_expert_prob = torch.max(expert_probs)
			plausible_tokens_mask = expert_probs >= (alpha * max_expert_prob)

			# Compute contrastive logits as the difference in log probabilities
			expert_log_probs = torch.log(expert_probs + 1e-10)  # Add small epsilon to avoid log(0)
			amateur_log_probs = torch.log(amateur_probs + 1e-10)
			contrastive_scores = expert_log_probs - amateur_log_probs

			# Set scores for implausible tokens to -inf
			contrastive_scores[~plausible_tokens_mask] = float("-inf")

			next_token_id = torch.argmax(contrastive_scores, dim=-1, keepdim=True)
			generated_tokens = torch.cat([generated_tokens, next_token_id], dim=1)

			# Break if EOS token is generated
			if next_token_id.item() == tokenizer.eos_token_id:
				break

	# Decode the token ids to string while skipping special tokens
	return tokenizer.decode(generated_tokens[0][len(input_ids):], skip_special_tokens=True)


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
    device = "mps" if torch.mps.is_available() else "cpu"
    amateur_model = tr.AutoModelForCausalLM.from_pretrained(amateur_path).to(device)
    expert_model = tr.AutoModelForCausalLM.from_pretrained(expert_path).to(device)
    print("Generating output...\n")
    output = contrastive_generation(amateur_model, expert_model, prompt, max_tokens=20)
    print("Generated Text:\n", output)


if __name__ == "__main__":
    main()
