# GPT2WithExtraAttentions

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class GPT2WithExtraAttentions(nn.Module):
    def __init__(self, model_name="gpt2", insert_every=2):
        super().__init__()
        # Load GPT-2 with its LM head
        self.gpt2 = AutoModelForCausalLM.from_pretrained(model_name)

        # Freeze all GPT-2 params
        for p in self.gpt2.parameters():
            p.requires_grad = False

        hidden_size = self.gpt2.config.hidden_size
        num_heads = self.gpt2.config.n_head

        # Add trainable attention layers after some GPT-2 blocks
        self.extra_attns = nn.ModuleDict()
        for i, block in enumerate(self.gpt2.transformer.h):
            if i % insert_every == 0:  # e.g. every 2 blocks
                self.extra_attns[str(i)] = nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    batch_first=True
                )

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Use GPT-2 embeddings + position encodings
        inputs_embeds = self.gpt2.transformer.wte(input_ids) + self.gpt2.transformer.wpe(
            torch.arange(input_ids.size(1), device=input_ids.device)
        )
        hidden_states = inputs_embeds

        # Forward through GPT-2 blocks + our adapters
        for i, block in enumerate(self.gpt2.transformer.h):
            hidden_states = block(hidden_states, attention_mask=attention_mask)[0]
            if str(i) in self.extra_attns:
                attn_out, _ = self.extra_attns[str(i)](
                    hidden_states, hidden_states, hidden_states
                )
                hidden_states = hidden_states + attn_out  # residual add

        # Final norm + LM head
        hidden_states = self.gpt2.transformer.ln_f(hidden_states)
        logits = self.gpt2.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return {"loss": loss, "logits": logits}
```
