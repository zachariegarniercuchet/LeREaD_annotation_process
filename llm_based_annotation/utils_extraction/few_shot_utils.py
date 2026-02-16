from utils_extraction.html_cleaner import clean_tokens
from utils_extraction.tokenizer_utils import decode
from utils_extraction.htmlLabel import HTMLLabel

def prepare_label_tokens(chunk, label_config):
    """
    Transform label tokens according to specified parameters.
    
    Handles both opening and closing tags, using a stack to track label names
    for proper closing tag generation in simplified mode.
    
    Args:
        chunk: List of tokens to process
        remove_attributes: List of attribute names to remove from labels
        keep_attributes: List of attribute names to keep in labels
        switch_type: If True, switch between manual_label and auto_label types
        use_simplified: If True, output simplified form (e.g., <title> instead of <manual_label labelname="title">)
        keep_labels: List of label names to keep (all others removed). If None, keep all labels.
        remove_labels: List of label names to remove. If None, no removal filtering.
    
    Returns:
        List of transformed tokens
    """

    # Get config parameters
    remove_attributes = label_config.get('remove_attributes', None)
    keep_attributes = label_config.get('keep_attributes', None)
    switch_type = label_config.get('switch_type', False)
    use_simplified = label_config.get('use_simplified', False)
    keep_labels = label_config.get('keep_labels', None)
    remove_labels = label_config.get('remove_labels', None)


    transformed_chunk = []
    label_name_stack = []  # Stack to track opened label names for closing tags
    skip_stack = []  # Stack to track which labels are being skipped (removed)
    
    for token in chunk:
        # Check if token is a label opening tag
        is_manual_open = token.lower().startswith('<manual_label') and token.endswith('>')
        is_auto_open = token.lower().startswith('<auto_label') and token.endswith('>')
        
        # Check if token is a label closing tag
        is_manual_close = token.lower().startswith('</manual_label') and token.endswith('>')
        is_auto_close = token.lower().startswith('</auto_label') and token.endswith('>')
        
        if is_manual_open or is_auto_open:
            # Process opening tag
            try:
                label = HTMLLabel(token)
                
                # Check if this label should be kept based on keep_labels/remove_labels
                should_keep = True
                if keep_labels is not None:
                    should_keep = label.name in keep_labels
                elif remove_labels is not None:
                    should_keep = label.name not in remove_labels
                
                if not should_keep:
                    # Skip this label tag, but track it for closing tag
                    skip_stack.append(True)
                    # Don't add to transformed_chunk (removes the tag but keeps content)
                    continue
                
                skip_stack.append(False)
                
                # Apply attribute filtering if specified
                if keep_attributes is not None or remove_attributes is not None:
                    label.to_string(remove_attributes=remove_attributes, keep_attributes=keep_attributes)
                
                # Switch type if requested
                if switch_type:
                    label.switch_type()
                
                # Output simplified or full form
                if use_simplified:
                    transformed_token = label.to_simplified()
                    # Push label name to stack for closing tag
                    label_name_stack.append(label.name)
                else:
                    transformed_token = str(label)
                
                transformed_chunk.append(transformed_token)
            except ValueError:
                # If token can't be parsed, keep as-is
                transformed_chunk.append(token)
                
        elif is_manual_close or is_auto_close:
            # Process closing tag
            # Check if we skipped the corresponding opening tag
            if skip_stack:
                was_skipped = skip_stack.pop()
                if was_skipped:
                    # Skip this closing tag too (removes the tag but keeps content)
                    continue
            
            if use_simplified:
                # Pop label name from stack
                if label_name_stack:
                    label_name = label_name_stack.pop()
                    transformed_token = f'</{label_name}>'
                else:
                    # Stack empty, keep as-is (shouldn't happen with valid HTML)
                    transformed_token = token
            else:
                # Not simplified: handle type switching if needed
                if switch_type:
                    # Switch closing tag type
                    if is_manual_close:
                        transformed_token = '</auto_label>'
                    else:
                        transformed_token = '</manual_label>'
                else:
                    transformed_token = token
            
            transformed_chunk.append(transformed_token)
        else:
            # Not a label tag, keep as-is
            transformed_chunk.append(token)
    
    return transformed_chunk


def extract_few_shot_examples(token_chunks, label_config):
    """
    Extract few-shot examples from the chunks with flexible label transformation.
    
    Args:
        token_chunks: List of token chunks to process
        remove_attributes: List of attribute names to remove from labels. If None, no removal filtering.
        keep_attributes: List of attribute names to keep in labels (all others removed). If None, no keep filtering.
        switch_type: If True, switch between manual_label and auto_label types
        use_simplified: If True, output simplified form (e.g., <title> instead of <manual_label labelname="title">)
        keep_labels: List of label names to keep (all others removed). If None, keep all labels.
        remove_labels: List of label names to remove. If None, no removal filtering.
    
    Returns:
        list: List of tuples (input_tokens, output_tokens)
              - input_tokens: cleaned tokens without any label tags
              - output_tokens: tokens with transformed label tags according to parameters
    
    Examples:
        >>> # Remove specific attributes and switch to auto_label
        >>> examples = extract_few_shot_examples(chunks, remove_attributes=['style', 'parent'], switch_type=True)
        
        >>> # Keep only labelname and docid attributes
        >>> examples = extract_few_shot_examples(chunks, keep_attributes=['labelname', 'docid'])
        
        >>> # Output simplified form with proper closing tags
        >>> examples = extract_few_shot_examples(chunks, use_simplified=True)
        >>> # <manual_label labelname="title">Text</manual_label> → <title>Text</title>
        
        >>> # Keep only decision labels (removes title tags but keeps text)
        >>> examples = extract_few_shot_examples(chunks, keep_labels=['decision'], use_simplified=True)
        >>> # <decision><title>John Campbell Law Corporation v.</title></decision>
        >>> # → <decision>John Campbell Law Corporation v.</decision>
        
        >>> # Remove title labels (keeps all other labels)
        >>> examples = extract_few_shot_examples(chunks, remove_labels=['title'], use_simplified=True)
        >>> # Same result as above
    """
    examples = []
    
    for chunk in token_chunks:
        # Input: tokens cleaned from all label tags
        input_chunk = clean_tokens(chunk, normalize=True, keep_manual_label=False, 
                                   keep_auto_label=False, keep_bookmarks=False)
        
        # Output: tokens with transformed label tags
        output_chunk = prepare_label_tokens(
            chunk, 
            label_config
        )
        
        examples.append((decode(input_chunk), decode(output_chunk)))
    
    print(f"   ✓ Extracted {len(examples)} few-shot examples from chunks")
    return examples


def select_few_shot(examples, n):
    """
    Select n few-shot examples from the provided list.
    
    Args:
        examples: List of tuples (input, expected_output)
        n: Number of examples to select
    Returns:
        list: Selected few-shot examples
    """
    if n >= len(examples):
        return examples
    else:
        return examples[:n]