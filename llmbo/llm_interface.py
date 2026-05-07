import json
import re
import os
from typing import List, Tuple
try:
    import anthropic
except ImportError:
    anthropic = None


class LLMInterface:
    """LLM API wrapper using Anthropic-compatible SDK (supports DeepSeek / Qwen)."""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]")
        api_key = api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
        if anthropic is None:
            raise ImportError("anthropic package required: pip install anthropic")
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    def chat(self, system: str, user_message: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return message.content[0].text

    def generate_candidates(
        self,
        history: List[Tuple[List[float], float]],
        param_bounds: List[List[float]],
        n_candidates: int = 3,
    ) -> List[List[float]]:
        prompt = self._build_candidate_prompt(history, param_bounds, n_candidates)
        system = (
            "You are an expert in analog circuit design and Bayesian optimization. "
            "Given the optimization history (parameter values -> Figure of Merit), "
            "propose promising parameter values to try next. "
            "Exploit areas of good performance while exploring uncertain regions. "
            "Respond ONLY with valid JSON: {\"candidates\": [[...], [...], ...], \"reasoning\": \"...\"}"
        )
        response = self.chat(system, prompt, temperature=0.7)
        return self._parse_candidates(response, len(param_bounds))

    def _build_candidate_prompt(
        self,
        history: List[Tuple[List[float], float]],
        param_bounds: List[List[float]],
        n_candidates: int,
    ) -> str:
        lines = ["Optimization history (parameters -> Figure of Merit):"]
        for i, (params, fom) in enumerate(history):
            lines.append(f"  Iter {i}: params={[round(p, 4) for p in params]} -> FOM={fom:.6f}")
        sorted_hist = sorted(history, key=lambda x: x[1], reverse=True)
        best_params, best_fom = sorted_hist[0]
        lines.append(f"\nBest so far: params={[round(p, 4) for p in best_params]} -> FOM={best_fom:.6f}")
        lines.append(f"\nParameter bounds: {[[round(b[0], 4), round(b[1], 4)] for b in param_bounds]}")
        lines.append(
            f"\nPropose {n_candidates} new parameter vectors to try next. "
            f"Consider both exploitation (near best point) and exploration (uncertain regions)."
        )
        return "\n".join(lines)

    def _parse_candidates(self, response: str, n_dims: int) -> List[List[float]]:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Cannot parse LLM response: {response[:300]}...")
        candidates = data.get("candidates", data.get("parameters", []))
        if not candidates:
            raise ValueError(f"No candidates found in response: {response[:300]}...")
        return [[float(v) for v in c[:n_dims]] for c in candidates[:n_dims]]
