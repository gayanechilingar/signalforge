"""Ollama provider — local open-source models.

This is the default backend: free, private, and good enough for most extraction
tasks once the prompt does the heavy lifting. It is also what makes the
model bake-off in ``evals/`` honest — swapping llama3.2:3b for llama3.1:8b is a
config change, not a rewrite.

Cost is genuinely $0, but we still report a *shadow* cost derived from
``configs/models.yaml`` so the accounting path is exercised in every run rather
than only when a paid provider is wired in.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from ..settings import get_settings
from .base import LLMClient, LLMError, LLMRequest, LLMResponse, Usage


class OllamaClient(LLMClient):
    provider = "ollama"

    def __init__(self, host: str | None = None, timeout_s: float | None = None) -> None:
        s = get_settings()
        self.host = (host or s.ollama_host).rstrip("/")
        self.timeout_s = timeout_s or s.ollama_timeout_s
        self._client = httpx.Client(base_url=self.host, timeout=self.timeout_s)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- api ---------------------------------------------------------------
    def complete(self, req: LLMRequest) -> LLMResponse:
        body: dict[str, Any] = {
            "model": req.model,
            "stream": False,
            "messages": [self._msg(m) for m in req.messages],
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens,
            },
        }
        if req.stop:
            body["options"]["stop"] = req.stop
        if req.json_mode:
            # Ollama accepts either "json" (well-formed) or a JSON Schema object
            # (grammar-constrained). Prefer the schema when we have one: newer
            # Ollama builds honour it, older ones fall back to plain JSON mode.
            body["format"] = req.json_schema or "json"
        if req.tools:
            body["tools"] = req.tools

        t0 = time.perf_counter()
        try:
            resp = self._client.post("/api/chat", json=body)
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama transport error: {exc}", provider=self.provider) from exc
        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code >= 400:
            detail = resp.text[:400]
            # A missing model is a config error; retrying will never fix it.
            retryable = resp.status_code >= 500 and "not found" not in detail.lower()
            raise LLMError(
                f"ollama {resp.status_code}: {detail}",
                provider=self.provider,
                retryable=retryable,
            )

        data = resp.json()
        if data.get("format_error"):  # defensive: schema-constrained decode failed
            raise LLMError(f"ollama format error: {data['format_error']}", provider=self.provider)

        msg = data.get("message") or {}
        usage = Usage(
            tokens_in=int(data.get("prompt_eval_count") or 0),
            tokens_out=int(data.get("eval_count") or 0),
        )
        return LLMResponse(
            text=msg.get("content") or "",
            model=req.model,
            provider=self.provider,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=data.get("done_reason"),
            tool_calls=self._tool_calls(msg),
            raw={k: v for k, v in data.items() if k != "message"},
        )

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        model = model or get_settings().embed_model or "nomic-embed-text"
        out: list[list[float]] = []
        # /api/embed handles batches on recent builds; /api/embeddings is the
        # single-input legacy route. Try batch first, degrade gracefully.
        try:
            resp = self._client.post("/api/embed", json={"model": model, "input": texts})
            if resp.status_code < 400:
                payload = resp.json()
                if payload.get("embeddings"):
                    return [list(map(float, e)) for e in payload["embeddings"]]
        except httpx.HTTPError:
            pass

        for text in texts:
            try:
                resp = self._client.post("/api/embeddings", json={"model": model, "prompt": text})
            except httpx.HTTPError as exc:
                raise LLMError(f"ollama embed error: {exc}", provider=self.provider) from exc
            if resp.status_code >= 400:
                raise LLMError(
                    f"ollama embed {resp.status_code}: {resp.text[:200]}", provider=self.provider
                )
            out.append([float(x) for x in resp.json()["embedding"]])
        return out

    def health(self) -> tuple[bool, str]:
        try:
            resp = self._client.get("/api/tags", timeout=5.0)
        except httpx.HTTPError as exc:
            return False, f"ollama unreachable at {self.host}: {exc}"
        if resp.status_code >= 400:
            return False, f"ollama {resp.status_code}"
        names = [m["name"] for m in resp.json().get("models", [])]
        return True, f"ollama ok ({len(names)} models: {', '.join(names[:4])})"

    def list_models(self) -> list[str]:
        resp = self._client.get("/api/tags")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _msg(m: Any) -> dict[str, Any]:
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = m.tool_calls
        return d

    @staticmethod
    def _tool_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalise Ollama's tool-call shape onto our internal one."""
        calls = msg.get("tool_calls") or []
        out = []
        for i, c in enumerate(calls):
            fn = c.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            out.append(
                {"id": c.get("id") or f"call_{i}", "name": fn.get("name"), "args": args or {}}
            )
        return out
