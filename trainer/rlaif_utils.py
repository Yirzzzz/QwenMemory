import json
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM


def parse_prompt_messages(prompt_messages_json: str, prompt_fallback: str) -> List[Dict[str, str]]:
    if prompt_messages_json:
        try:
            parsed = json.loads(prompt_messages_json)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return [{"role": "user", "content": prompt_fallback}]


def score_with_reward_model(reward_model, reward_tokenizer, messages, device, max_length=4096):
    if hasattr(reward_model, 'get_score'):
        return float(reward_model.get_score(reward_tokenizer, messages))

    text = None
    if hasattr(reward_tokenizer, 'apply_chat_template'):
        try:
            text = reward_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = None

    if text is None:
        text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])

    inputs = reward_tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = reward_model(**inputs)
        logits = getattr(outputs, 'logits', None)
        if logits is None:
            raise ValueError('Reward model does not expose get_score or logits.')
        return float(logits.reshape(-1)[0].item())


def get_per_token_logps(model, input_ids, n_keep):
    outputs = model(input_ids)
    logits = outputs.logits[:, -(n_keep + 1):-1, :]
    target_ids = input_ids[:, -n_keep:]
    return F.log_softmax(logits, dim=-1).gather(2, target_ids.unsqueeze(-1)).squeeze(-1)


def build_aux_loss(outputs, device):
    aux = getattr(outputs, 'aux_loss', None)
    return aux if aux is not None else torch.tensor(0.0, device=device)


class AutoCausalLMCritic(nn.Module):
    def __init__(self, hf_model_path, hidden_size, trust_remote_code=True):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(hf_model_path, trust_remote_code=trust_remote_code)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            **kwargs,
        )
        hidden_states = outputs.hidden_states[-1]
        return self.value_head(hidden_states).squeeze(-1)
