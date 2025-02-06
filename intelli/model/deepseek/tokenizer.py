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
        model_file = os.path.join(model_path, "tokenizer.model")
        
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
        elif os.path.exists(model_file):
            # Try loading binary model
            self.sp_model.Load(model_file)
        else:
            # Create a basic tokenizer from scratch
            print("No tokenizer files found, creating basic tokenizer...")
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                # Write basic vocabulary with special tokens first
                basic_vocab = [
                    "<|endoftext|>",
                    "<|user|>",
                    "<|assistant|>",
                    "<|system|>",
                    "<s>",
                    "</s>",
                    "<pad>",
                    ".", ",", "!", "?", "-", "'", '"', "\n",
                    # Programming-specific tokens
                    "def", "class", "return", "import", "from", "if", "else", "for", "while",
                    "try", "except", "raise", "with", "as", "in", "is", "not", "and", "or",
                    "True", "False", "None", "self", "__init__", "print", "range", "len",
                    # Add basic characters
                    *[chr(i) for i in range(ord('a'), ord('z')+1)],  # a-z
                    *[chr(i) for i in range(ord('A'), ord('Z')+1)],  # A-Z
                    *[chr(i) for i in range(ord('0'), ord('9')+1)],  # 0-9
                    *[chr(i) for i in range(0x4E00, 0x9FFF)]  # Common Chinese characters
                ]
                for token in basic_vocab:
                    f.write(f"{token}\n")
                f.flush()
                
                # Train basic model with improved parameters
                spm.SentencePieceTrainer.Train(
                    f'--input={f.name} '
                    f'--model_prefix={model_file[:-6]} '
                    '--vocab_size=32000 '
                    '--character_coverage=0.9995 '
                    '--model_type=unigram '
                    '--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 '
                    '--control_symbols=<|user|>,<|assistant|>,<|system|> '
                    '--user_defined_symbols=<s>,</s>,<pad> '
                    '--treat_whitespace_as_suffix=true '
                    '--remove_extra_whitespaces=false '
                    '--byte_fallback=true '
                    '--normalization_rule_name=identity '
                    '--add_dummy_prefix=false '
                    '--max_sentence_length=8192'
                )
                
                # Clean up
                os.unlink(f.name)
            
            # Load the created model
            self.sp_model.Load(model_file)
        
        # Load config if available
        config_file = os.path.join(model_path, "tokenizer_config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # Set special tokens with defaults
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.system_token = "<|system|>"
        
        # Get token IDs
        self.pad_token_id = self.sp_model.piece_to_id(self.pad_token)
        self.bos_token_id = self.sp_model.piece_to_id(self.bos_token)
        self.eos_token_id = self.sp_model.piece_to_id(self.eos_token)
        self.user_token_id = self.sp_model.piece_to_id(self.user_token)
        self.assistant_token_id = self.sp_model.piece_to_id(self.assistant_token)
        self.system_token_id = self.sp_model.piece_to_id(self.system_token)
        
        # Set chat template with improved formatting
        self.chat_template = self.config.get('chat_template', """
{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
{{ '<|system|>\n' + message['content'] + '\n' }}
{%- elif message['role'] == 'user' -%}
{{ '<|user|>\n' + message['content'] + '\n' }}
{%- elif message['role'] == 'assistant' -%}
{{ '<|assistant|>\n' + message['content'] + '\n' }}
{%- endif -%}
{%- endfor -%}
<|assistant|>
""".strip())
    
    def _write_spm_model(self, output_path: str):
        """Write a SentencePiece model file from JSON vocab."""
        # Create a temporary text file with the vocabulary and sample text
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as vocab_file:
            # Write each token on a new line with its score and sample text
            vocab = self.tokenizer_json.get('vocab', {})
            
            # Write sample text for training
            sample_text = "This is a sample text to ensure the model has some training data.\n"
            sample_text += "Hello world! How are you doing today?\n"
            sample_text += "The quick brown fox jumps over the lazy dog.\n"
            vocab_file.write(sample_text)
            
            # Write vocabulary items
            for token, _ in sorted(vocab.items(), key=lambda x: x[1]):
                # Add the token as a sample text as well
                if len(token) > 0:  # Skip empty tokens
                    try:
                        # Try to decode the token if it's a byte sequence
                        token_text = bytes([int(token[2:], 16)]).decode('utf-8') if token.startswith('0x') else token
                        vocab_file.write(f"{token_text}\n")
                    except:
                        # If decoding fails, write the token as is
                        vocab_file.write(f"{token}\n")
            vocab_file.flush()
            
            # Train a new SentencePiece model
            import sentencepiece as spm
            spm.SentencePieceTrainer.Train(
                f'--input={vocab_file.name} '
                f'--model_prefix={output_path[:-6]} '  # Remove .model suffix
                '--vocab_size=8000 '  # Reduced vocab size to match available tokens
                '--character_coverage=1.0 '
                '--model_type=unigram '
                '--pad_id=0 '
                '--bos_id=1 '
                '--eos_id=2 '
                '--unk_id=3 '
                '--input_format=text '  # Changed to text format
                '--hard_vocab_limit=false '
                '--normalization_rule_name=identity '
                '--treat_whitespace_as_suffix=true '
                '--add_dummy_prefix=false '
                '--remove_extra_whitespaces=false '
                '--max_sentence_length=8192 '
                '--split_by_unicode_script=false '
                '--split_by_whitespace=false '
                '--split_digits=false '
                '--byte_fallback=true'  # Enable byte fallback for unknown characters
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
            # Define special tokens to skip
            special_tokens = {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.user_token_id,
                self.assistant_token_id,
                self.system_token_id
            }
            # Filter out special tokens
            ids = [id for id in ids if id not in special_tokens and id != -1]
        
        # Decode remaining tokens
        text = self.sp_model.decode(ids)
        
        # Clean up any remaining special token text
        if skip_special_tokens:
            special_strings = ['<|endoftext|>', '<|user|>', '<|assistant|>', '<|system|>']
            for s in special_strings:
                text = text.replace(s, '')
        
        return text.strip()

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