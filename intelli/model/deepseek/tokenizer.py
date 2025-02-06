import os
import json
from typing import List, Optional
import sentencepiece as spm


class DeepSeekTokenizer:
    """Tokenizer for DeepSeek models."""
    
    def __init__(self, model_path: str):
        """
        Initialize tokenizer.
        
        Args:
            model_path: Path to model directory containing tokenizer files
        """
        self.sp_model = spm.SentencePieceProcessor()
        
        # Try loading JSON tokenizer first
        json_file = os.path.join(model_path, "tokenizer.json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                self.tokenizer_json = json.load(f)
            # Create a temporary SPM model from the vocab
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.model', delete=False) as tmp:
                # Write SPM model in binary format
                self._write_spm_model(tmp.name)
                # Load the temporary model
                self.sp_model.Load(tmp.name)
                # Clean up
                os.unlink(tmp.name)
        else:
            # Fall back to binary model
            vocab_file = os.path.join(model_path, "tokenizer.model")
            if not os.path.exists(vocab_file):
                raise FileNotFoundError(f"No tokenizer found at {model_path}")
            self.sp_model.Load(vocab_file)
        
        # Load config if available
        config_file = os.path.join(model_path, "tokenizer_config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # Set special tokens
        self.pad_token_id = self.config.get('pad_token_id', 0)
        self.eos_token_id = self.config.get('eos_token_id', 2)
        self.bos_token_id = self.config.get('bos_token_id', 1)
    
    def _write_spm_model(self, output_path: str):
        """Write a SentencePiece model file from JSON vocab."""
        # Basic SPM model structure
        model_proto = {
            'pieces': [],
            'trainer_spec': {
                'vocab_size': len(self.tokenizer_json.get('vocab', {})),
                'character_coverage': 1.0,
                'model_type': 'BPE',
                'input_format': 'piece',
                'hard_vocab_limit': False,
                'pad_id': self.tokenizer_json.get('pad_token_id', 0),
                'bos_id': self.tokenizer_json.get('bos_token_id', 1),
                'eos_id': self.tokenizer_json.get('eos_token_id', 2),
                'unk_id': self.tokenizer_json.get('unk_token_id', 3),
            }
        }
        
        # Add vocab pieces
        vocab = self.tokenizer_json.get('vocab', {})
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            model_proto['pieces'].append({
                'piece': token,
                'score': 0.0,
                'type': 'NORMAL'
            })
        
        # Write binary model file
        import struct
        with open(output_path, 'wb') as f:
            # Write magic number and version
            f.write(b'\x01\x02\x03\x04\x05\x06\x07\x08')
            f.write(struct.pack('<Q', 0x0000000000000001))
            
            # Write model proto
            proto_bytes = str(model_proto).encode('utf-8')
            f.write(struct.pack('<Q', len(proto_bytes)))
            f.write(proto_bytes)
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to encode
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
            
        Returns:
            List of token IDs
        """
        ids = self.sp_model.encode(text)
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to remove special tokens from output
            
        Returns:
            Decoded text
        """
        if skip_special_tokens:
            ids = [id for id in ids if id not in {self.pad_token_id, self.bos_token_id, self.eos_token_id}]
        return self.sp_model.decode(ids)

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