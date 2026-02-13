import tiktoken
from typing import List, Dict, Any

# Standard pricing for common models (per 1M tokens) - Update as needed
PRICING = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4o": {"input": 5.00, "output": 15.00},
    "openai/gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    # Add others as needed
}

def get_token_count(text: str, model_name: str = "gpt-4o") -> int:
    """
    Returns the number of tokens in a text string.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def get_accurate_cost(sources: List[Dict], query: str, answer: str, prompt_template: Any, model_name: str = "openai/gpt-4o-mini") -> float:
    """
    Calculates the cost of a RAG generation.
    Args:
        sources: List of source documents (dicts or objects)
        query: User query
        answer: Generated answer
        prompt_template: LangChain prompt template used
        model_name: Model identifier for pricing
    Returns:
        float: Estimated cost in USD
    """
    # 1. Format context string from sources
    context_parts = []
    for s in sources:
        # Handle both dict and object formats
        if isinstance(s, dict):
            content = s.get('doc', {}).page_content if isinstance(s.get('doc'), object) else s.get('page_content', '')
        else:
            content = s.page_content if hasattr(s, 'page_content') else str(s)
        context_parts.append(content)
    
    context_str = "\n\n".join(context_parts)

    # 2. Construct full prompt (approximation)
    # If prompt_template is a LangChain object, try to format it
    try:
        if hasattr(prompt_template, "format"):
            full_prompt = prompt_template.format(context=context_str, question=query)
        else:
            # Fallback manual construction
            full_prompt = f"Context: {context_str}\n\nQuestion: {query}"
    except Exception:
        full_prompt = f"Context: {context_str}\n\nQuestion: {query}"

    # 3. Count tokens
    input_tokens = get_token_count(full_prompt)
    output_tokens = get_token_count(answer)

    # 4. Calculate cost
    # Default to gpt-4o-mini pricing if model not found
    prices = PRICING.get(model_name) or PRICING.get("openai/gpt-4o-mini")
    
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]

    return round(input_cost + output_cost, 6)
