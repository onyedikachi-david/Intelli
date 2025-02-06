import os
import json
from typing import List, Optional, Dict
import sentencepiece as spm


class DeepSeekTokenizer:
    """Lightweight tokenizer for DeepSeek models using SentencePiece."""
    
    def __init__(self, model_path: str):
        """
        Initialize tokenizer from model path.
        
        Args:
            model_path: Path to model directory containing tokenizer files
        """
        # Load vocabulary and merge rules
        vocab_file = os.path.join(model_path, "tokenizer.model")
        config_file = os.path.join(model_path, "tokenizer_config.json")
        
        if not os.path.exists(vocab_file):
            raise ValueError(f"Tokenizer model file not found at {vocab_file}")
            
        # Load SentencePiece model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        
        # Load config if available
        self.config = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        
        # Special tokens
        self.bos_token_id = self.config.get('bos_token_id', 1)
        self.eos_token_id = self.config.get('eos_token_id', 2)
        self.pad_token_id = self.config.get('pad_token_id', 0)
        
        # Chat template
        self.chat_template = self.config.get('chat_template', "{%- for message in messages -%}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + '\n<|assistant|>\n' }}\n{% elif message['role'] == 'assistant' %}\n{{ message['content'] + '\n' }}\n{% endif %}\n{%- endfor -%}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        return self.sp_model.EncodeAsIds(text)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids to text."""
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in {self.bos_token_id, self.eos_token_id, self.pad_token_id}]
        return self.sp_model.DecodeIds(token_ids)
    
    def apply_chat_template(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> List[int]:
        """Apply chat template to format messages."""
        from jinja2 import Template
        
        # Format messages using template
        template = Template(self.chat_template)
        formatted = template.render(messages=messages)
        
        # Encode and add special tokens
        tokens = self.encode(formatted)
        if add_generation_prompt:
            tokens = [self.bos_token_id] + tokens
            
        return tokens 