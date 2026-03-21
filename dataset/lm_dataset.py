from torch.utils.data import Dataset
import torch
import os
import random
import json
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    system_prompts = [
        "You are Qwen, a helpful AI assistant.",
        "You are a careful assistant. Provide clear and accurate answers.",
        "You are a reliable multilingual assistant.",
        "你是 Qwen，一个有帮助、谨慎且准确的助手。",
        "你是一个专业 AI 助手，请尽量给出准确且有条理的回答。",
    ]
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(system_prompts)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = conversations.copy()
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels_legacy(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def encode_with_assistant_mask(self, conversations):
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        encoded = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            tools=tools
        )
        input_ids = encoded["input_ids"]
        assistant_mask = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))
        if assistant_mask is None:
            raise KeyError("assistant mask is missing")
        labels = [tok if int(mask) == 1 else -100 for tok, mask in zip(input_ids, assistant_mask)]
        return input_ids, labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])

        try:
            input_ids, labels = self.encode_with_assistant_mask(conversations)
        except Exception:
            prompt = self.create_chat_prompt(conversations)
            prompt = post_processing_chat(prompt)
            input_ids = self.tokenizer(prompt).input_ids
            labels = self.generate_labels_legacy(input_ids)

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels += [-100] * (self.max_length - len(labels))
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def encode_chat_and_mask(self, messages):
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True
            )
            input_ids = encoded["input_ids"][:self.max_length]
            assistant_mask = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))
            if assistant_mask is None:
                raise KeyError("assistant mask is missing")
            assistant_mask = [int(v) for v in assistant_mask[:self.max_length]]
            pad_len = self.max_length - len(input_ids)
            input_ids += [self.padding] * pad_len
            assistant_mask += [0] * pad_len
            return input_ids, assistant_mask
        except Exception:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt = post_processing_chat(prompt)
            encoding = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding='max_length')
            input_ids = encoding['input_ids']
            loss_mask = self.generate_loss_mask_legacy(input_ids)
            return input_ids, loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']
        rejected = sample['rejected']

        chosen_input_ids, chosen_loss_mask = self.encode_chat_and_mask(chosen)
        rejected_input_ids, rejected_loss_mask = self.encode_chat_and_mask(rejected)

        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask_legacy(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']

        prompt_messages = messages[:-1]
        prompt = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt = post_processing_chat(prompt)
        return prompt, answer, prompt_messages

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer, prompt_messages = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer,
            'prompt_messages_json': json.dumps(prompt_messages, ensure_ascii=False)
        }


if __name__ == "__main__":
    pass
