"""
LLM and Embedding Factory Functions
Centralized factory functions to avoid code duplication across notebooks.
"""
import os
from typing import Dict, Any

def get_llm(config: Dict[str, Any]):
    """
    Return a LangChain-compatible chat model based on config.
    Supports fallback if enable_fallback is True.
    Args:
        config: Configuration dictionary with llm_provider, llm_model, etc.   
    Returns:
        LangChain chat model instance (potentially with fallbacks)
    """
    primary_llm = _create_llm_instance(config, config["llm_provider"], config.get("llm_model"))
    
    if config.get("enable_fallback", False):
        fallback_provider = config.get("fallback_provider")
        fallback_model = config.get("fallback_model")
        
        if fallback_provider and fallback_model:
            # Create a localized config for the fallback to avoid mutating the original
            fallback_config = config.copy()
            fallback_config["llm_provider"] = fallback_provider
            fallback_config["llm_model"] = fallback_model
            
            secondary_llm = _create_llm_instance(fallback_config, fallback_provider, fallback_model)
            return primary_llm.with_fallbacks([secondary_llm])
            
    return primary_llm


def _create_llm_instance(config: Dict[str, Any], provider: str, model_name: str):
    """
    Internal helper to create a single LLM instance.
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            timeout=config["request_timeout"],
        )
    
    elif provider == "openrouter":
        # Construct full model name: provider/model
        openrouter_provider = config.get("openrouter_provider", "openai")
        openrouter_model = config.get("openrouter_model", "gpt-4o-mini")
        # If model_name is passed explicitly (e.g. for fallback), use it, otherwise construct
        # actually for openrouter primary, we construct it. 
        # But if this function is called for a fallback that is NOT openrouter, this block won't run.
        # If fallback IS openrouter, we might need to handle it, but for now fallback is likely Groq.
        
        # Make sure we use the constructed name only if we define it here, 
        # but for simplicity let's rely on config if it's the primary.
        if config.get("llm_provider") == "openrouter" and provider == "openrouter":
             full_model_name = f"{openrouter_provider}/{openrouter_model}"
        else:
             full_model_name = model_name

        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=full_model_name,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            timeout=config["request_timeout"],
        )
    
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            timeout=config["request_timeout"],
        )
    
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=config["temperature"],
            max_output_tokens=config["max_tokens"],
        )
    
    elif provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            model=model_name,
            temperature=config["temperature"],
        )
    
    elif provider == "hf_local":
        raise NotImplementedError(
            "HF local requires custom wrapper. "
            "Use HuggingFacePipeline or HuggingFaceEndpoint."
        )
    
    else:
        raise ValueError(f"Unknown llm_provider: {provider}")


# LangChain Embeddings Factory
def get_text_embeddings(config: Dict[str, Any]):
    """
    Return LangChain embeddings based on config.
    Args:
        config: Configuration dictionary with text_emb_provider, text_emb_model, etc.
    Returns:
        LangChain embeddings instance
    """
    provider = config["text_emb_provider"]
    model_name = config["text_emb_model"]
    
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_name)
    
    elif provider == "cohere":
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings(model=model_name)
    
    elif provider == "sbert":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_kwargs = {"device": "cpu"} #cuda or mps
        
        if config.get("normalize_embeddings", True):
            encode_kwargs = {"normalize_embeddings": True}
        else:
            encode_kwargs = {}
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    
    else:
        raise ValueError(f"Unknown text_emb_provider: {provider}")


def get_clip_model(config: Dict[str, Any]):
    """
    Return SentenceTransformer CLIP model for image+text embeddings.
    Args:
        config: Configuration dictionary with clip_model setting
    Returns:
        SentenceTransformer CLIP model instance
    """
    from sentence_transformers import SentenceTransformer
    clip_model_name = config.get("clip_model", "clip-ViT-B-32")
    return SentenceTransformer(clip_model_name)


# LlamaIndex LLM Factory
def get_llamaindex_llm(config: Dict[str, Any]):
    """
    Return LlamaIndex LLM based on config.
    Args:
        config: Configuration dictionary with llm_provider, llm_model, etc.
    Returns:
        LlamaIndex LLM instance
    """
    provider = config["llm_provider"]
    
    if provider == "openai":
        model_name = config["llm_model"]
        try:
            from llama_index_llms_openai import OpenAI
        except ImportError:
            from llama_index.llms.openai import OpenAI
        
        return OpenAI(
            model=model_name,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )
    
    elif provider == "openrouter":
        # Construct full model name: provider/model
        openrouter_provider = config.get("openrouter_provider", "openai")
        openrouter_model = config.get("openrouter_model", "gpt-4o-mini")
        model_name = f"{openrouter_provider}/{openrouter_model}"
        
        # LlamaIndex's OpenAI class validates model names, which fails with OpenRouter
        # Use OpenAILike which is designed for OpenAI-compatible APIs
        try:
            # Try the newer import structure first
            from llama_index_llms_openai_like import OpenAILike
            
            return OpenAILike(
                api_base="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                model=model_name,
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                is_chat_model=True,
            )
        except ImportError:
            try:
                # Try the old import structure
                from llama_index.llms.openai_like import OpenAILike
                
                return OpenAILike(
                    api_base="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    model=model_name,
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"],
                    is_chat_model=True,
                )
            except ImportError:
                # Final fallback: Use OpenAI with base_url
                from llama_index_llms_openai import OpenAI as LlamaOpenAI
                
                return LlamaOpenAI(
                    api_base="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    model=model_name,
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"],
                )
    
    elif provider == "groq":
        model_name = config["llm_model"]
        try:
            from llama_index_llms_groq import Groq
        except ImportError:
            from llama_index.llms.groq import Groq
        
        return Groq(
            model=model_name,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )
    
    elif provider == "gemini":
        model_name = config["llm_model"]
        try:
            from llama_index_llms_gemini import Gemini
        except ImportError:
            from llama_index.llms.gemini import Gemini
        
        return Gemini(
            model=model_name,
            temperature=config["temperature"],
        )
    
    else:
        # Fallback to LangChain adapter
        model_name = config["llm_model"]
        from langchain_openai import ChatOpenAI
        from llama_index.core.llms import LangChainLLM
        lc_llm = ChatOpenAI(
            model=model_name,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
        )
        return LangChainLLM(llm=lc_llm)


# LlamaIndex Embeddings Factory
def get_llamaindex_embeddings(config: Dict[str, Any]):
    """
    Return LlamaIndex embeddings based on config.
    Args:
        config: Configuration dictionary with text_emb_provider, text_emb_model, etc.
    Returns:
        LlamaIndex embeddings instance
    """
    provider = config["text_emb_provider"]
    model_name = config["text_emb_model"]
    
    if provider == "openai":
        try:
            from llama_index_embeddings_openai import OpenAIEmbedding
        except ImportError:
            from llama_index.embeddings.openai import OpenAIEmbedding
        
        return OpenAIEmbedding(model=model_name)
    
    elif provider == "cohere":
        try:
            from llama_index_embeddings_cohere import CohereEmbedding
        except ImportError:
            from llama_index.embeddings.cohere import CohereEmbedding
        
        return CohereEmbedding(model_name=model_name)
    
    elif provider == "sbert":
        try:
            from llama_index_embeddings_huggingface import HuggingFaceEmbedding
        except ImportError:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        return HuggingFaceEmbedding(
            model_name=model_name,
            device="cpu",
        )
    
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# Document Parser Factory
def get_pdf_parser(config: Dict[str, Any]):
    """
    Return a parser function or object based on config.
    """
    provider = config.get("parsing_provider", "pypdf")
    
    if provider == "llama_parse":
        try:
            from llama_parse import LlamaParse
        except ImportError:
            raise ImportError("llama-parse not installed. Run `pip install llama-parse`.")
        
        return LlamaParse(
            api_key=os.getenv("LLAMA_INDEX_API_KEY"), # User uses this env var
            result_type="markdown",
            verbose=True,
            language="en",
        )
            
    elif provider == "pypdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader
        
    else:
        raise ValueError(f"Unknown parsing provider: {provider}")


def load_pdf_and_save(pdf_path: str, parser: Any, output_dir: str = None) -> str:
    """
    Load PDF using the provided parser and save the full text to a markdown file.
    Args:
        pdf_path: Path to the PDF file
        parser: The parser instance (LlamaParse) or class (PyPDFLoader)
        output_dir: Directory to save the parsed markdown file. If None, uses PDF's parent.
    Returns:
        Full text content of the PDF
    """
    from pathlib import Path
    
    print(f"Loading PDF from: {pdf_path}")
    
    # Handle different parser types
    if hasattr(parser, "load_data"):
        # LlamaParse
        documents = parser.load_data(str(pdf_path))
    else:
        # PyPDFLoader (LangChain loader class)
        loader = parser(str(pdf_path))
        documents = loader.load()
        
    print(f"Loaded {len(documents)} pages/documents.")
    
    # Combine into full text
    full_text = "\n\n".join([doc.page_content if hasattr(doc, "page_content") else doc.text for doc in documents])
    
    # Save to file
    pdf_path_obj = Path(pdf_path)
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        save_path = out_path / f"{pdf_path_obj.stem}_parsed.md"
    else:
        save_path = pdf_path_obj.parent / f"{pdf_path_obj.stem}_parsed.md"
        
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(full_text)
        
    print(f"Saved parsed content to: {save_path}")
    return full_text


# Utility Functions
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    Args:
        config_path: Path to config.yaml file
    Returns:
        Configuration dictionary
    """
    import yaml
    from pathlib import Path
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Please ensure config.yaml exists in the project root."
        )
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def validate_api_keys(config: Dict[str, Any], verbose: bool = True) -> Dict[str, bool]:
    """
    Check which API keys are available in the environment.
    Args:
        config: Configuration dictionary
        verbose: If True, print warnings for missing keys
    Returns:
        Dictionary mapping key names to availability (True/False)
    """
    import warnings
    
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
        "LLAMA_INDEX_API_KEY": os.getenv("LLAMA_INDEX_API_KEY"),
    }    
    availability = {}
    for key, value in api_keys.items():
        availability[key] = value is not None
        if verbose and not value:
            warnings.warn(f"  {key} not found in environment")
    
    return availability


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the configuration.
    Args:
        config: Configuration dictionary
    """
    print(" Config loaded:")
    
    # Display LLM info with OpenRouter-specific formatting
    if config['llm_provider'] == 'openrouter':
        openrouter_provider = config.get('openrouter_provider', 'openai')
        openrouter_model = config.get('openrouter_model', 'gpt-4o-mini')
        print(f"  LLM: {config['llm_provider']} ({openrouter_provider}/{openrouter_model})")
    else:
        print(f"  LLM: {config['llm_provider']} / {config['llm_model']}")
    
    print(f"  Embeddings: {config['text_emb_provider']} / {config['text_emb_model']}")
    print(f"  Temperature: {config['temperature']}")
    print(f"  Artifacts: {config['artifacts_root']}")


# Example Usage
if __name__ == "__main__":
    """
    Example usage of the factory functions.
    Run this script to test: python llm_services.py
    """
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Load config
    config = load_config("config.yaml")
    
    # Print summary
    print_config_summary(config)
    print()
    
    # Validate API keys
    print(" API Key Status:")
    availability = validate_api_keys(config, verbose=False)
    for key, available in availability.items():
        status = "Working" if available else "Not Working"
        print(f"  {status} {key}")
    print()
    
    # Initialize LangChain LLM
    try:
        llm = get_llm(config)
        print(f" LangChain LLM initialized: {config['llm_provider']}")
    except Exception as e:
        print(f" LangChain LLM failed: {e}")
    
    # Initialize LangChain embeddings
    try:
        embeddings = get_text_embeddings(config)
        print(f" LangChain embeddings initialized: {config['text_emb_provider']}")
    except Exception as e:
        print(f" LangChain embeddings failed: {e}")
    
    print("\n All services initialized successfully!")

