"""LLM-guided prompt construction for DP-TIG: p = G_LLM(c, s)  (paper Sec. 3.1.2, Eq. 2, Fig. 3).

The instruction text follows Fig. 3. Generated prompts are validated against the paper's constraints
(starts with "A photo of", under 12 words, contains the target category and the descriptor) and
regenerated with a different sampling seed on failure. Every result is cached on disk keyed by
(category, descriptor, seed), so DP-TIG runs are reproducible and the LLM is not needed on the GPU box.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol, Tuple

INSTRUCTION = (
    'You are a prompt generator for a text-to-image model. Generate exactly one short prompt beginning with '
    '"A photo of a ...".\n'
    "Rules:\n"
    "- Use one target category and the specified conditioning descriptor.\n"
    "- Keep the prompt under 12 words.\n"
    "- Include both the target category and descriptor.\n"
    "- Do not add adjectives, objects, or background elements.\n"
    "- Output only the prompt.\n"
    "General structure: A photo of a [target category] [conditioning descriptor]"
)


def user_message(category: str, descriptor_text: str) -> str:
    return f"Target category: {category}\nConditioning descriptor: {descriptor_text}"


class LLMBackend(Protocol):
    def complete(self, instruction: str, user: str, seed: int) -> str: ...


class LlamaCppBackend:
    """Llama-3-8B-Instruct-32k-v0.1 Q4_K_M GGUF via llama-cpp-python (paper Sec. 5.4.1)."""

    def __init__(self, model_path: str, temperature: float = 0.5, top_p: float = 0.95,
                 n_ctx: int = 2048, n_gpu_layers: int = -1, max_tokens: int = 40):
        from llama_cpp import Llama  # lazy: only needed when actually generating
        self.llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens

    def complete(self, instruction: str, user: str, seed: int) -> str:
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": instruction}, {"role": "user", "content": user}],
            temperature=self.temperature, top_p=self.top_p, seed=seed, max_tokens=self.max_tokens)
        return out["choices"][0]["message"]["content"]


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def clean(raw: str) -> str:
    """First non-empty line, surrounding quotes/whitespace removed."""
    for line in raw.strip().splitlines():
        line = line.strip().strip("\"'`“”").strip()
        if line:
            return line
    return ""


def validate_prompt(prompt: str, category: str, keyword: str, max_words: int = 12) -> Tuple[bool, str]:
    """Paper constraints. Returns (ok, reason). 'Under 12 words' => at most 11."""
    if not prompt:
        return False, "empty"
    if not prompt.lower().startswith("a photo of"):
        return False, "must begin with 'A photo of'"
    if len(prompt.split()) >= max_words:
        return False, f"not under {max_words} words"
    low = prompt.lower()
    if category.lower() not in low:
        return False, "missing target category"
    if keyword.lower() not in low:
        return False, "missing conditioning descriptor"
    return True, "ok"


def fallback_prompt(category: str, descriptor_text: str) -> str:
    """Deterministic template used only if the LLM never satisfies the constraints (flagged in metadata)."""
    return f"A photo of {_article(category)} {category} {descriptor_text}".strip()


@dataclass
class GeneratedPrompt:
    prompt: str
    attempts: int
    fallback: bool


class PromptGenerator:
    def __init__(self, backend: Optional[LLMBackend], cache_path: Optional[Path] = None,
                 max_retries: int = 5, max_words: int = 12):
        self.backend, self.max_retries, self.max_words = backend, max_retries, max_words
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: Dict[str, GeneratedPrompt] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._cache[r["key"]] = GeneratedPrompt(r["prompt"], r["attempts"], r["fallback"])

    @staticmethod
    def _key(category: str, keyword: str, seed: int) -> str:
        return f"{category}|{keyword}|{seed}"

    def generate(self, category: str, keyword: str, seed: int, descriptor_text: Optional[str] = None) -> GeneratedPrompt:
        """keyword: the descriptor term that must appear (e.g. 'plate'); descriptor_text: what the LLM is
        shown (e.g. 'on a plate'; defaults to keyword)."""
        text = descriptor_text or keyword
        key = self._key(category, keyword, seed)
        if key in self._cache:
            return self._cache[key]
        if self.backend is None:
            raise RuntimeError(f"prompt not in cache and no LLM backend loaded: {key}")
        result = None
        for attempt in range(self.max_retries):
            raw = self.backend.complete(INSTRUCTION, user_message(category, text), seed + attempt * 1_000_003)
            p = clean(raw)
            ok, _ = validate_prompt(p, category, keyword, self.max_words)
            if ok:
                result = GeneratedPrompt(p, attempt + 1, False)
                break
        if result is None:
            result = GeneratedPrompt(fallback_prompt(category, text), self.max_retries, True)
        self._cache[key] = result
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as f:
                f.write(json.dumps(dict(key=key, category=category, descriptor=keyword, seed=seed,
                                        prompt=result.prompt, attempts=result.attempts,
                                        fallback=result.fallback)) + "\n")
        return result


class LazyBackend:
    """Defers loading the (multi-GB) model until a prompt is actually missing from the cache."""

    def __init__(self, factory):
        self._factory, self._backend = factory, None

    def complete(self, instruction: str, user: str, seed: int) -> str:
        if self._backend is None:
            self._backend = self._factory()
        return self._backend.complete(instruction, user, seed)


def build_prompt_generator(llm_cfg: dict, cache_path: Path, model_override: Optional[str] = None) -> PromptGenerator:
    from .config import resolve_weight
    model = resolve_weight(model_override or llm_cfg["model"])
    backend = LazyBackend(lambda: LlamaCppBackend(model, temperature=llm_cfg.get("temperature", 0.5),
                                                  top_p=llm_cfg.get("top_p", 0.95)))
    return PromptGenerator(backend, cache_path, max_retries=llm_cfg.get("max_retries", 5),
                           max_words=llm_cfg.get("max_words", 12))
