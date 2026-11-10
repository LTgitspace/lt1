import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"]


class NanoTokenizer:
    """
    Byte-level BPE tokenizer wrapper supporting special instruction/chat tokens.
    """
    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer
        self.pad_id = self._tok.token_to_id("<|pad|>")
        self.bos_id = self._tok.token_to_id("<|bos|>")
        self.eos_id = self._tok.token_to_id("<|eos|>")
        self.user_id = self._tok.token_to_id("<|user|>")
        self.assistant_id = self._tok.token_to_id("<|assistant|>")

    @classmethod
    def train_from_iterator(
        cls,
        iterator,
        vocab_size: int = 16384,
        save_path: str | None = None,
    ) -> "NanoTokenizer":
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(iterator, trainer=trainer)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            tokenizer.save(save_path)

        return cls(tokenizer)

    @classmethod
    def load(cls, path: str) -> "NanoTokenizer":
        tokenizer = Tokenizer.from_file(path)
        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()
