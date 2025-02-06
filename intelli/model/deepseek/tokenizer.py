import os
import json
from typing import List, Optional, Dict
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
        
        # Set chat template
        self.chat_template = self.config.get('chat_template', "{%- for message in messages -%}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + '\n<|assistant|>\n' }}\n{% elif message['role'] == 'assistant' %}\n{{ message['content'] + '\n' }}\n{% endif %}\n{%- endfor -%}")
    
    def _write_spm_model(self, output_path: str):
        """Write a SentencePiece model file from JSON vocab."""
        # Create a temporary text file with the vocabulary
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as vocab_file:
            # Write each token on a new line with its score
            vocab = self.tokenizer_json.get('vocab', {})
            for token, _ in sorted(vocab.items(), key=lambda x: x[1]):
                # Escape special characters in the token
                escaped_token = token.encode('unicode_escape').decode('utf-8')
                vocab_file.write(f"{escaped_token}\t0.0\n")
            vocab_file.flush()
            
            # Train a new SentencePiece model
            import sentencepiece as spm
            spm.SentencePieceTrainer.Train(
                f'--input={vocab_file.name} '
                f'--model_prefix={output_path[:-6]} '  # Remove .model suffix
                '--vocab_size=32000 '  # Large enough for most vocabularies
                '--character_coverage=1.0 '
                '--model_type=bpe '
                '--pad_id=0 '
                '--bos_id=1 '
                '--eos_id=2 '
                '--unk_id=3 '
                '--input_format=tsv '
                '--hard_vocab_limit=false '
                '--normalization_rule_name=identity '
                '--treat_whitespace_as_suffix=true '
                '--add_dummy_prefix=false '
                '--remove_extra_whitespaces=false'
            )
            
            # Clean up
            os.unlink(vocab_file.name)
            
            # Move the trained model to the desired location
            import shutil
            shutil.move(f"{output_path[:-6]}.model", output_path)
            # Clean up the extra files
            if os.path.exists(f"{output_path[:-6]}.vocab"):
                os.unlink(f"{output_path[:-6]}.vocab")
    
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