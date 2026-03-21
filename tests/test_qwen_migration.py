import unittest
import torch

from trainer.trainer_utils import build_ckpt_name
from trainer.rlaif_utils import get_per_token_logps


class DummyConfig:
    hidden_size = 768
    use_moe = False


class ToyModel(torch.nn.Module):
    def __init__(self, vocab=8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, 4)
        self.head = torch.nn.Linear(4, vocab)

    def forward(self, input_ids):
        h = self.emb(input_ids)
        logits = self.head(h)
        return type("Obj", (), {"logits": logits})()


class TestQwenMigrationHelpers(unittest.TestCase):
    def test_build_ckpt_name(self):
        self.assertEqual(build_ckpt_name("full_sft", DummyConfig(), None), "full_sft_768")
        self.assertEqual(build_ckpt_name("full_sft", DummyConfig(), "qwen"), "full_sft_qwen")

    def test_per_token_logps_shape(self):
        model = ToyModel(vocab=16)
        input_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
        logps = get_per_token_logps(model, input_ids, n_keep=2)
        self.assertEqual(tuple(logps.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
