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
        else:
            # Initialize local model
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            logger.info(f"Loading local model: {model_name}")
            logger.info("This may take a few minutes on first run...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with GPU support
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
                low_cpu_mem_usage=True
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
            if self.provider in ["openai", "nvidia"]:
                # OpenAI or NVIDIA Build API
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