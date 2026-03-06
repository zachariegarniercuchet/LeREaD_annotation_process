import itertools
import re 

def chunk_tokens(tokens: list, min_tokens: int = 500, stop_bookmark_separation=False) -> list:
    """
    Chunk a list of tokens into chunks of at least min_tokens.
    
    STRICT RULES:
    1. Accumulate tokens until reaching min_tokens
    2. After min_tokens, continue ONLY until:
        - Quotations are closed (not inside_quotes)
        - AND <manual_label> tags are closed (depth == 0)
        - AND current token is a word (not whitespace/punctuation)
    
    If stop_bookmark_separation=True, splits at bookmark first, then chunks each part separately.
    
    Returns list of token chunks, or tuple of two lists if bookmark separation.
    """
    
    # If stop_bookmark_separation is True, find and split at bookmark
    if stop_bookmark_separation:
        bookmark_idx = None
        for i in range(len(tokens) - 2):
            if (tokens[i] == '<htmllabelizer_bookmark id="stop">' and 
                tokens[i+1] == '🔖' and 
                tokens[i+2] == '</htmllabelizer_bookmark>'):
                bookmark_idx = i
                break
        
        if bookmark_idx is not None:
            # Split tokens into before and after (excluding the 3 bookmark tokens)
            tokens_before = tokens[:bookmark_idx]
            tokens_after = tokens[bookmark_idx + 3:]
            
            print(f"   ✓ Found bookmark separator at index {bookmark_idx}")
            print(f"   ✓ Splitting: {len(tokens_before)} tokens before, {len(tokens_after)} tokens after")
            
            # Chunk both parts separately
            chunks_before = _chunk_token_list(tokens_before, min_tokens)
            chunks_after = _chunk_token_list(tokens_after, min_tokens)
            
            print(f"   ✓ Total chunks: {len(chunks_before)} before + {len(chunks_after)} after = {len(chunks_before) + len(chunks_after)}")
            return chunks_before, chunks_after
        else:
            print(f"   ⚠ Warning: stop_bookmark_separation=True but bookmark not found")
    
    # Default behavior: chunk entire list
    return _chunk_token_list(tokens, min_tokens)

def _chunk_token_list(tokens: list, min_tokens: int) -> list:
    """
    Internal method to chunk a token list.
    
    Algorithm:
    1. Accumulate tokens until count >= min_tokens
    2. After min_tokens, continue until ALL conditions are met:
        - NOT inside quotes
        - manual_label depth == 0
        - current token is a word
    """
    chunks = []
    current = []
    count = 0
    
    inside_quotes = False
    manual_label_depth = 0
    
    def is_word_token(tok: str) -> bool:
        """Check if token is a word (alphanumeric, >=1 char)."""
        return tok and tok.strip() and re.match(r'^\w+$', tok) is not None
    
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        current.append(tok)
        count += 1
        
        # Track quote state - ONLY opening quote " starts, ONLY closing quote " ends
        if tok == '“':  # U+201C - LEFT DOUBLE QUOTATION MARK (opening)
            inside_quotes = True
        elif tok == '”':  # U+201D - RIGHT DOUBLE QUOTATION MARK (closing)
            inside_quotes = False
        
        # Track manual_label tag depth
        if tok.startswith('<manual_label'):
            manual_label_depth += 1
        elif tok.startswith('</manual_label'):
            manual_label_depth -= 1
        
        # Check if we can end the chunk
        if count >= min_tokens:
            # Check ALL conditions:
            # 1. Not inside quotes
            # 2. manual_label depth is 0
            # 3. Current token is a word
            if not inside_quotes and manual_label_depth == 0 and is_word_token(tok):
                # All conditions met: end chunk here
                chunks.append(current)
                current = []
                count = 0
                # Reset states for safety
                inside_quotes = False
                manual_label_depth = 0
        
        i += 1
    
    # Append remaining tokens as final chunk
    if current:
        chunks.append(current)
    
    print(f"   ✓ Chunked tokens into {len(chunks)} chunks (>= {min_tokens} tokens each)")
    return chunks




def flatten_token_chunks(token_chunks: list[list[str]], separator: str = None) -> list[str]:
    """
    Flatten a list of token chunks (list of lists) back into a single token list.
    Preserves token order exactly.
    
    Args:
        token_chunks: List of token chunks to flatten
        separator: Optional separator token to insert between chunks (e.g., '<sep>')
    """
    if separator:
        flat = []
        for i, chunk in enumerate(token_chunks):
            flat.extend(chunk)
            if i < len(token_chunks) - 1:  # Don't add separator after last chunk
                flat.append(separator)
    else:
        flat = list(itertools.chain.from_iterable(token_chunks))
    
    print(f"   ✓ Flattened {len(token_chunks)} chunks into {len(flat)} tokens")
    return flat