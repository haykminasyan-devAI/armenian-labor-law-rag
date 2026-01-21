# LLM generator
"""
LLM generator for RAG pipeline - Supports OpenAI and local HuggingFace models.
"""

import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LLMGenerator:
    """Generate answers using OpenAI or local HuggingFace LLM."""
    
    def __init__(self, 
                 model_name: str = "gpt-4-turbo",
                 provider: str = "openai",  # "openai" or "local"
                 api_key: Optional[str] = None,
                 device: str = "auto",
                 max_tokens: int = 1000,
                 temperature: float = 0.1):
        """
        Initialize LLM generator.
        
        Args:
            model_name: Model name (OpenAI model or HuggingFace model)
            provider: "openai" or "local"
            api_key: API key for OpenAI (or set OPENAI_API_KEY env var)
            device: Device for local models ("auto", "cuda", "cpu")
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        if provider == "openai":
            # Initialize OpenAI
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
            logger.info(f"✅ Initialized OpenAI client with model: {model_name}")
        
        elif provider == "nvidia":
            # Initialize NVIDIA Build API (OpenAI-compatible)
            from openai import OpenAI
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key or os.getenv('NVIDIA_API_KEY')
            )
            logger.info(f"✅ Initialized NVIDIA Build client with model: {model_name}")
        
        elif provider == "groq":
            # Initialize Groq API (ultra-fast LPU inference)
            from openai import OpenAI
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key or os.getenv('GROQ_API_KEY')
            )
            logger.info(f"✅ Initialized Groq client with model: {model_name}")
        
        elif provider == "google":
            # Initialize Google Gemini
            import google.generativeai as genai
            genai.configure(api_key=api_key or os.getenv('GOOGLE_API_KEY'))
            self.genai = genai
            self.gemini_model = genai.GenerativeModel(model_name)
            logger.info(f"✅ Initialized Google Gemini with model: {model_name}")
        else:
            # Initialize local model
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import os
            import warnings
            
            # Completely disable progress bars and logging for headless mode
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
            warnings.filterwarnings('ignore')
            
            # Disable transformers logging
            import transformers
            transformers.logging.set_verbosity_error()
            
            logger.info(f"Loading local model: {model_name}")
            
            # Check GPU availability
            if torch.cuda.is_available():
                logger.info("✅ GPU detected! Loading model on GPU...")
                device = "cuda:0"
                device_map = "auto"
            else:
                logger.warning("⚠️  No GPU detected! Using CPU (will be slow)...")
                device = "cpu"
                device_map = None
            
            logger.info("This may take a few minutes on first run...")
            
            # Load tokenizer (quietly)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with GPU support (completely silent)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device_map if device_map else device,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            
            # Create pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                top_p=0.9
            )
            
            device_str = str(next(self.model.parameters()).device)
            logger.info(f"✅ Model loaded successfully on {device_str}")
            if torch.cuda.is_available():
                logger.info(f"   GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments
            
        Returns:
            Generated text
        """
        try:
            if self.provider == "google":
                # Google Gemini API
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={
                        'max_output_tokens': kwargs.get('max_tokens', self.max_tokens),
                        'temperature': kwargs.get('temperature', self.temperature)
                    }
                )
                answer = response.text.strip()
                
            elif self.provider in ["openai", "nvidia", "groq"]:
                # OpenAI, NVIDIA Build, or Groq API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=kwargs.get('max_tokens', self.max_tokens),
                    temperature=kwargs.get('temperature', self.temperature)
                )
                answer = response.choices[0].message.content.strip()
                
            else:
                # Local HuggingFace model
                messages = [
                    {"role": "system", "content": "You are a helpful legal assistant specialized in Armenian Labor Law."},
                    {"role": "user", "content": prompt}
                ]
                
                # Apply chat template if available
                if self.tokenizer.chat_template:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                else:
                    formatted_prompt = f"System: You are a helpful legal assistant.\n\nUser: {prompt}\n\nAssistant:"
                
                # Generate
                outputs = self.pipe(
                    formatted_prompt,
                    max_new_tokens=kwargs.get('max_tokens', self.max_tokens),
                    temperature=kwargs.get('temperature', self.temperature),
                    return_full_text=False
                )
                
                answer = outputs[0]['generated_text'].strip()
            
            logger.debug(f"Generated answer: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def __del__(self):
        """Clean up GPU memory."""
        if self.provider == "local" and hasattr(self, 'model'):
            import torch
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()