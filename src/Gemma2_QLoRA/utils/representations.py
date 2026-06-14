import torch


def pooled_last_hidden(outputs, attention_mask: torch.Tensor) -> torch.Tensor:
    hidden_states = outputs.hidden_states[-1]
    sequence_lengths = attention_mask.sum(dim=1).to(hidden_states.device) - 1
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, sequence_lengths]
