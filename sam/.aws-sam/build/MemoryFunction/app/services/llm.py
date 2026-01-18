"""
LLM Service Module - Supports OpenAI GPT-4o and Google Gemini.
Provides unified interface for text generation and embeddings.

Model Hierarchy (accuracy-first):
1. gpt-4o - Most capable, best for complex political analysis
2. gpt-4o-mini - Fast, cost-effective for simpler tasks
3. gpt-4-turbo - Legacy, still very capable
4. gemini-1.5-pro - Google alternative
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
import re
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
import httpx


# Advanced model options
ADVANCED_MODELS = {
    "gpt-4o": "Most capable, best accuracy for complex analysis",
    "gpt-4o-2024-08-06": "Latest GPT-4o with structured outputs",
    "gpt-4o-mini": "Fast and cost-effective, good for simpler queries",
    "gpt-4-turbo": "Previous generation, still very capable",
    "gpt-4-turbo-2024-04-09": "Latest GPT-4 Turbo",
}

# Default system prompt for political analysis
POLITICAL_ANALYST_SYSTEM_PROMPT = """You are an expert political strategist and electoral analyst specializing in Indian politics, particularly West Bengal elections.

Your core principles:
1. ACCURACY FIRST: Only state facts you can verify from provided data
2. NO HALLUCINATION: If data is unavailable, say so explicitly
3. EVIDENCE-BASED: Every claim must reference specific data points
4. QUANTITATIVE: Use numbers, percentages, and statistics
5. ACTIONABLE: Provide specific, implementable recommendations

Your expertise includes:
- Electoral data analysis (seat projections, margins, swings)
- Voter segmentation (demographics, persuadability, turnout)
- Campaign strategy (ground game, messaging, resource allocation)
- Opposition research (vulnerabilities, counter-strategies)
- Regional politics (constituency-level dynamics, local issues)

Response guidelines:
- Use markdown formatting for clarity
- Include tables for comparative data
- Cite sources with [Source: filename/datapoint]
- Rate confidence as High/Medium/Low with explanation
- Provide both quantitative analysis and strategic recommendations"""


@dataclass
class LLMResponse:
    """Wrapper for LLM response."""
    text: str
    usage: Dict[str, int] = None
    model: str = ""


class BaseLLM:
    """Base LLM interface."""
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        response_format: Optional[str] = None
    ) -> LLMResponse:
        raise NotImplementedError
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError
    
    def extract_json(self, text: str) -> Any:
        """Extract JSON from LLM response text."""
        # Try to find JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON array
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        # Return raw text if no JSON found
        return text
    
    @property
    def political_system_prompt(self) -> str:
        """Get the default political analyst system prompt."""
        return POLITICAL_ANALYST_SYSTEM_PROMPT


class OpenAILLM(BaseLLM):
    """OpenAI GPT-4o implementation - Most advanced model for best accuracy."""
    
    # Model upgrade mapping - always use most capable
    MODEL_UPGRADES = {
        "gpt-4-turbo": "gpt-4o",
        "gpt-4-turbo-preview": "gpt-4o",
        "gpt-4": "gpt-4o",
        "gpt-3.5-turbo": "gpt-4o-mini",
    }
    
    def __init__(self, force_model: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.openai_api_key)
        
        # Use forced model or upgrade from settings
        requested_model = force_model or settings.openai_model
        self.requested_model = requested_model
        
        # Automatically upgrade to more capable model
        if requested_model in self.MODEL_UPGRADES:
            self.model = self.MODEL_UPGRADES[requested_model]
            print(f"[LLM] Upgraded model: {requested_model} -> {self.model}")
        else:
            self.model = requested_model
        
        # Use best embedding model
        self.embed_model = settings.openai_embed_model or "text-embedding-3-large"
        # Some newer models use max_completion_tokens instead of max_tokens (e.g., GPT-5 family)
        self._use_max_completion_tokens = str(self.model).startswith("gpt-5")

    def _chat_completions_http(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call OpenAI Chat Completions via raw HTTP.

        Why: older openai-python versions may not support newer parameters like
        `max_completion_tokens`, while the API does.
        """
        if not settings.openai_api_key:
            raise RuntimeError("OpenAI API key not configured")

        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _is_model_not_found_error(exc: Exception) -> bool:
        """
        Detect OpenAI 'model not found / not available / not supported on this endpoint' errors
        across library versions.
        """
        name = exc.__class__.__name__
        if name in {"NotFoundError"}:
            return True
        msg = str(exc).lower()
        if "model" not in msg:
            return False

        # Common availability/endpoint-support signals (MODEL-related, not parameter-related)
        needles = (
            "not found",
            "does not exist",
            "no such model",
            "not available",
            "use the responses api",
            "use responses api",
            "chat/completions",
            "v1/chat/completions",
        )
        return any(n in msg for n in needles)

    @staticmethod
    def _is_max_tokens_unsupported_error(exc: Exception) -> bool:
        """
        Detect OpenAI errors indicating that `max_tokens` is unsupported and
        `max_completion_tokens` should be used instead (some newer models).
        """
        msg = str(exc).lower()
        return (
            "max_tokens" in msg
            and (
                "max_completion_tokens" in msg
                or "max_output_tokens" in msg  # some providers/models use this wording
                or "unsupported_parameter" in msg
            )
            and ("unsupported" in msg or "not supported" in msg)
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
        use_political_system: bool = False
    ) -> LLMResponse:
        if temperature is None:
            temperature = settings.openai_temperature
        
        # Use political analyst system prompt if requested and no custom system provided
        if use_political_system and not system:
            system = self.political_system_prompt
            
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # Build a request payload we can use for either openai-python or raw HTTP
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        # Token limit parameter differs by model family
        if self._use_max_completion_tokens:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        
        # GPT-4o supports structured outputs better
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        
        try:
            # For models requiring max_completion_tokens, use raw HTTP to avoid
            # older SDK incompatibilities.
            if self._use_max_completion_tokens:
                data = self._chat_completions_http(payload)
                return LLMResponse(
                    text=data["choices"][0]["message"]["content"],
                    usage={
                        "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                    },
                    model=data.get("model", self.model),
                )

            # Default path: openai-python
            response = self.client.chat.completions.create(**payload)
        except Exception as e:
            # Always log the error message (helps debug bad request vs rate limit vs model support issues)
            print(f"[LLM] OpenAI request error (model={self.model}): {e}")

            # Some models require `max_completion_tokens` instead of `max_tokens`
            if not self._use_max_completion_tokens and self._is_max_tokens_unsupported_error(e):
                print(f"[LLM] Retrying with max_completion_tokens for model '{self.model}'...")
                self._use_max_completion_tokens = True
                payload.pop("max_tokens", None)
                payload["max_completion_tokens"] = max_tokens

                data = self._chat_completions_http(payload)
                return LLMResponse(
                    text=data["choices"][0]["message"]["content"],
                    usage={
                        "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                    },
                    model=data.get("model", self.model),
                )

            # If user configured an unavailable model, fall back safely.
            if self._is_model_not_found_error(e) and self.model != "gpt-4o":
                print(
                    f"[LLM] WARNING: OpenAI model '{self.model}' not available. "
                    f"Falling back to 'gpt-4o'. Error: {e}"
                )
                self.model = "gpt-4o"
                # Reset token parameter mode when falling back
                self._use_max_completion_tokens = False
                payload["model"] = self.model
                payload.pop("max_completion_tokens", None)
                payload["max_tokens"] = max_tokens
                response = self.client.chat.completions.create(**payload)
            else:
                raise
        
        return LLMResponse(
            text=response.choices[0].message.content,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            model=response.model
        )
    
    def generate_with_context(
        self,
        prompt: str,
        context: List[str],
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """Generate with retrieved context for RAG."""
        context_text = "\n\n".join([f"[Context {i+1}]: {c}" for i, c in enumerate(context)])
        
        enhanced_prompt = f"""Based on the following verified information:

{context_text}

User Question: {prompt}

Instructions:
1. Answer ONLY based on the provided context
2. If the context doesn't contain the answer, say "I don't have data for this"
3. Cite specific context numbers [Context X] when making claims
4. Be precise with numbers and percentages"""

        return self.generate(
            enhanced_prompt, 
            system=system or self.political_system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed(self, texts: List[str]) -> List[List[float]]:
        # OpenAI has a limit on batch size, process in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.embed_model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings


class GeminiLLM(BaseLLM):
    """Google Gemini implementation - Using most advanced Gemini model."""
    
    # Prefer most capable Gemini models
    PREFERRED_MODELS = ["gemini-1.5-pro-latest", "gemini-1.5-pro", "gemini-1.5-flash"]
    
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        
        # Use the most advanced available model
        model_name = settings.gemini_model
        if model_name not in self.PREFERRED_MODELS:
            model_name = "gemini-1.5-pro"  # Default to most capable
        
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.embed_model_name = settings.gemini_embed_model
        self._genai = genai
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        response_format: Optional[str] = None,
        use_political_system: bool = False
    ) -> LLMResponse:
        # Use political analyst system prompt if requested
        if use_political_system and not system:
            system = self.political_system_prompt
        
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        if response_format == "json":
            generation_config["response_mime_type"] = "application/json"
        
        response = self.model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        return LLMResponse(
            text=response.text,
            usage={},
            model=self.model_name
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = self._genai.embed_content(
                model=f"models/{self.embed_model_name}",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
        return embeddings


class MockLLM(BaseLLM):
    """Mock LLM for testing without API calls."""
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        response_format: Optional[str] = None
    ) -> LLMResponse:
        # Generate mock responses based on prompt keywords
        if "sub-queries" in prompt.lower() or "sub queries" in prompt.lower():
            return LLMResponse(
                text='["voter demographics", "historical voting patterns", "local issues"]',
                model="mock"
            )
        if "swot" in prompt.lower():
            return LLMResponse(
                text='{"strengths": ["Strong local presence"], "weaknesses": ["Limited funding"], "opportunities": ["Youth engagement"], "threats": ["Opposition momentum"], "priority_actions": ["Increase ground presence"]}',
                model="mock"
            )
        return LLMResponse(
            text='{"answer": "This is a mock response for testing."}',
            model="mock"
        )
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        return [np.random.randn(768).tolist() for _ in texts]


@lru_cache(maxsize=1)
def get_llm(force_model: Optional[str] = None) -> BaseLLM:
    """
    Factory function to get configured LLM instance.
    
    Args:
        force_model: Override the model selection (e.g., 'gpt-4o', 'gpt-4o-mini')
    
    Returns:
        Configured LLM instance
    """
    provider = settings.llm_provider.lower()
    
    print(f"[LLM] Initializing LLM provider: {provider}")
    print(f"[LLM] OpenAI API Key: {'SET (' + settings.openai_api_key[:8] + '...)' if settings.openai_api_key else 'NOT SET'}")
    print(f"[LLM] Gemini API Key: {'SET' if settings.gemini_api_key else 'NOT SET'}")
    
    if provider == "openai" and settings.openai_api_key:
        llm = OpenAILLM(force_model=force_model)
        print(f"[LLM] Using OpenAI with model: {llm.model} (requested: {settings.openai_model})")
        return llm
    elif provider == "gemini" and settings.gemini_api_key:
        llm = GeminiLLM()
        print(f"[LLM] Using Gemini with model: {llm.model_name}")
        return llm
    elif settings.openai_api_key:
        llm = OpenAILLM(force_model=force_model)
        print(f"[LLM] Falling back to OpenAI with model: {llm.model}")
        return llm
    elif settings.gemini_api_key:
        llm = GeminiLLM()
        print(f"[LLM] Falling back to Gemini with model: {llm.model_name}")
        return llm
    else:
        print("[LLM] WARNING: No API keys found! Using MockLLM for testing.")
        print("[LLM] Set OPENAI_API_KEY or GEMINI_API_KEY in .env file for actual responses.")
        return MockLLM()


def get_advanced_llm() -> BaseLLM:
    """Get the most advanced LLM available - always uses gpt-4o for maximum accuracy."""
    get_llm.cache_clear()  # Clear cache to allow re-initialization
    return get_llm(force_model="gpt-4o")


def get_fast_llm() -> BaseLLM:
    """Get a fast LLM for simpler tasks - uses gpt-4o-mini."""
    return OpenAILLM(force_model="gpt-4o-mini") if settings.openai_api_key else get_llm()


# Export the enhanced system prompt for external use
def get_political_system_prompt() -> str:
    """Get the expert political analyst system prompt."""
    return POLITICAL_ANALYST_SYSTEM_PROMPT
