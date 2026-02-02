import re
from utils.html_utils import is_auto_label_tag, is_manual_label_tag, is_tag_token, strip_auto_labels
from utils.tokenizer_utils import tokenize
from utils.htmlLabel import from_simplified


def extract_start_end_tokens(tokens: list) -> list:
    """
    Extract tokens between <start> and <end> markers.
    
    Args:
        tokens: List of tokens containing <start> and <end> markers
    
    Returns:
        List of tokens between markers, or original tokens if markers not found
    
    Raises:
        ValueError: If <start> found but <end> not found, or vice versa
    """
    try:
        start_index = tokens.index("<start>")
    except ValueError:
        print("   ⚠ Warning: <start> marker not found, returning original tokens")
        return tokens
    
    try:
        end_index = tokens.index("<end>")
    except ValueError:
        raise ValueError("<start> marker found but <end> marker missing")
    
    if end_index <= start_index:
        raise ValueError("<end> marker appears before <start> marker")
    
    extracted = tokens[start_index + 1 : end_index]
    print(f"   ✓ Extracted {len(extracted)} tokens between <start> and <end>")
    return extracted


def simplified_to_normal_form(tokens: list, label_type: str = 'auto_label') -> list:
    """
    Convert simplified label format to normal auto_label or manual_label format.
    
    Transforms:
        <decision> → <auto_label labelname="decision">
        </decision> → </auto_label>
        <title titletype="main"> → <auto_label labelname="title" titletype="main">
        </title> → </auto_label>
    
    Args:
        tokens: List of tokens potentially containing simplified label tags
        label_type: Either 'auto_label' or 'manual_label' (default: 'auto_label')
    
    Returns:
        List of tokens with normalized label format
    """
    if label_type not in ['auto_label', 'manual_label']:
        raise ValueError(f"label_type must be 'auto_label' or 'manual_label', got: {label_type}")
    
    normalized_tokens = []
    
    for token in tokens:
        # Check for opening tag: <...> but not </...> or <manual_label...> or <auto_label...>
        is_open = bool(re.fullmatch(r'<(?!\/|manual_label|auto_label)[^>]+>', token))
        
        # Check for closing tag: </...> but not </manual_label...> or </auto_label...>
        is_close = bool(re.fullmatch(r'<\/((?!manual_label|auto_label)[^>]+)>', token))
        
        if is_open:
            # Convert simplified opening tag to normal form
            html_label = from_simplified(token, label_type=label_type)
            normalized_tokens.append(html_label._token)
        elif is_close:
            # Convert simplified closing tag to normal form
            normalized_tokens.append(f'</{label_type}>')
        else:
            # Keep token as-is
            normalized_tokens.append(token)
    
    print(f"   ✓ Converted {len(tokens)} tokens from simplified to {label_type} format")
    return normalized_tokens


def apply_post_processing_transforms(raw_output: str, use_simplified: bool = False, label_type: str = 'auto_label') -> list:
    """
    Apply all post-processing transformations to raw LLM output.
    
    Pipeline:
    1. Tokenize raw output
    2. Extract tokens between <start> and <end> markers
    3. Convert simplified format to normal form (if applicable)
    
    Args:
        raw_output: Raw text output from LLM
        use_simplified: Whether the LLM output uses simplified format
        label_type: Target label type ('auto_label' or 'manual_label')
    
    Returns:
        List of processed tokens ready for verification
    """
    # Step 1: Tokenize
    tokens = tokenize(raw_output)
    print(f"   → Step 1: Tokenized into {len(tokens)} tokens")
    
    # Step 2: Extract between <start> and <end>
    try:
        tokens = extract_start_end_tokens(tokens)
        print(f"   → Step 2: Extracted {len(tokens)} tokens between markers")
    except ValueError as e:
        print(f"   ⚠ Warning: {str(e)}")
    
    # Step 3: Convert simplified to normal form if needed
    if use_simplified:
        tokens = simplified_to_normal_form(tokens, label_type=label_type)
        print(f"   → Step 3: Converted to {label_type} format")
    
    return tokens


def _tokens_equivalent(tok1: str, tok2: str) -> bool:
        """
        Check if two tokens are equivalent.
        
        Tokens are equivalent if:
        1. They are exactly equal (tok1 == tok2)
        2. Both are pure whitespace (any combination of spaces, newlines, tabs, nbsp, etc.)
        3. Both are manual_label tags (opening or closing)
        
        Examples:
        - 'hello' == 'hello' -> True
        - 'hello' == 'world' -> False
        - ' ' == '\n\xa0\n\n\n\n' -> True (both are pure whitespace)
        - ',' == ',' -> True
        - ',' == ' ' -> False
        - '<manual_label labelname="mention">' == '<manual_label labelname="title">' -> True
        """
        # First check: exact equality
        if tok1 == tok2:
            return True
        
        # Second check: both are pure whitespace (different variants)
        if not tok1.strip() and not tok2.strip():
            return True
        
        # Third check: both are manual_label tags (opening or closing)
        
        if is_manual_label_tag(tok1) != 0 and is_manual_label_tag(tok2) != 0:
            return True
        
        # Otherwise, not equivalent
        return False


def merge_tokens_with_auto_labels(tokens: list[str], processed_tokens: list[str], output_dir=None, filename=None, log=False) -> list[str]:
    """
    Merge original tokens with processed tokens to produce the original text
    with only <auto_label ...> tags inserted from processed_tokens.

    Algorithm:
    - If tokens are equivalent (same or both whitespace): take original token, advance both indices
    - If tokens differ:
      - If processed token is a auto_label tag: it's an insertion, take it and advance idx2 only
      - Otherwise: take original token and advance idx1 only
    
    This handles insertions of <auto_label> tags in the processed version.
    """
    n1 = len(tokens)
    n2 = len(processed_tokens)
    result = []
    idx1 = 0
    idx2 = 0
    count = 0

    txt = ""

    while idx1 < n1 and idx2 < n2:
        t1 = tokens[idx1]
        t2 = processed_tokens[idx2]
        txt += f"TOKEN1 {t1}\n"
        txt += f"TOKEN2 {t2}\n"
        
        if _tokens_equivalent(t1, t2):
            # Tokens match: keep original and advance both
            result.append(t1)
            idx1 += 1
            idx2 += 1
            txt += f"COUNT {count}\n"
            
            count = 0
        else:
            count +=1
            # Tokens differ
            if is_auto_label_tag(t2) != 0: # meaning opening or closing auto tag

                if is_auto_label_tag(t2) == 1 : # Meaning opening auto tag
                    # Find the first non-tag word after the opening auto_label tag
                    next_idx = idx2 + 1
                    target_word = None
                    
                    # Look for the first non-tag token after the auto_label opening tag
                    while next_idx < n2:
                        next_token = processed_tokens[next_idx]
                        if not is_tag_token(next_token):
                            target_word = next_token
                            break
                        next_idx += 1
                    
                    if target_word is not None:
                        # Search forward in list1 to find the target word
                        search_idx = idx1
                        found = False
                        
                        while search_idx < n1:
                            if _tokens_equivalent(tokens[search_idx], target_word):
                                # Found the target word in list1
                                # Add all tokens from idx1 to search_idx (excluding search_idx)
                                while idx1 < search_idx:
                                    result.append(tokens[idx1])
                                    txt += f"PRE-TAG TOKEN: {tokens[idx1]}\n"
                                    idx1 += 1
                                
                                # Now add the opening auto_label tag
                                result.append(t2)
                                txt += f"OPENING AUTO_LABEL: {t2}\n"
                                idx2 += 1
                                
                                found = True
                                break
                            search_idx += 1
                        
                        if not found:
                            # If target word not found in list1, just add the auto_label tag
                            result.append(t2)
                            txt += f"AUTO_LABEL (no match found): {t2}\n"
                            idx2 += 1
                    else:
                        # No non-tag token found after auto_label, just add it
                        result.append(t2)
                        txt += f"AUTO_LABEL (no target): {t2}\n"
                        idx2 += 1
                        
                else : # Closing auto tag
                # t2 is a closing auto_label tag (insertion in processed version)
                    result.append(t2)
                    idx2 += 1
            else:
                # t2 is not a auto_label: keep original token
                result.append(t1)
                idx1 += 1
        txt += f"DECISION : {result[-1]}\n"

    # Append remaining tokens from original (if any)
    if idx1 < n1:
        txt += f"t1 {len(tokens[idx1:])}\n"
        result.extend(tokens[idx1:])
    
    # Append remaining tokens from processed (if any, likely closing tags)
    if idx2 < n2:
        txt += f"idx2 {idx2}\n"
        txt += f"{processed_tokens[idx2:]}\n"
        txt += f"t2 {len(processed_tokens[idx2:])}\n"
        result.extend(processed_tokens[idx2:])
    
    if output_dir and filename:
        with open(f"{output_dir}/debug_merge_{filename}.txt", "w", encoding="utf-8") as f:
            f.write(txt)

    if log:
        print(f"   ✓ Merged {len(tokens)} original and {len(processed_tokens)} processed into {len(result)} tokens")
    return result




def add_style_and_parent_to_auto_labels(html_content: str, label_scheme_path: str = None) -> str:
    """
    Add parent and style attributes to auto_label tags in HTML content based on label scheme JSON.
    
    The function loads the label scheme from a JSON file to determine:
    - Parent relationships (top-level labels have parent="", sublabels have parent="<parent_label>")
    - Colors for each label (converted to background-color style)
    
    Args:
        html_content: HTML string with auto_label tags
        label_scheme_path: Path to label_scheme.json file. If None, uses default path.
    
    Returns:
        Modified HTML string with parent and style attributes added
    """
    import re
    import json
    import os
    
    # Default path to label scheme
    if label_scheme_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        label_scheme_path = os.path.join(current_dir, '..', '..', 'ressources', 'label_scheme.json')
    
    # Load label scheme
    try:
        with open(label_scheme_path, 'r', encoding='utf-8') as f:
            label_scheme = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Label scheme file not found at {label_scheme_path}. Using auto_label tags without modification.")
        return html_content
    
    # Build parent and style mappings from label scheme
    parent_map = {}
    style_map = {}
    
    # Helper function to determine text color based on background brightness
    def get_text_color(hex_color: str) -> str:
        """Determine if text should be black or white based on background color brightness."""
        # Remove # if present
        hex_color = hex_color.lstrip('#')
        # Convert to RGB
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        # Calculate relative luminance
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return 'black' if luminance > 0.5 else 'white'
    
    # Convert hex color to rgb format
    def hex_to_rgb(hex_color: str) -> str:
        """Convert hex color to rgb() format."""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgb({r}, {g}, {b})"
    
    # Process label scheme to build mappings
    for parent_label, parent_data in label_scheme.items():
        # Top-level labels have no parent
        parent_map[parent_label] = ""
        
        # Set style for top-level label
        if 'color' in parent_data:
            bg_color = hex_to_rgb(parent_data['color'])
            text_color = get_text_color(parent_data['color'])
            style_map[parent_label] = f"background-color: {bg_color}; color: {text_color};"
        
        # Process sublabels
        if 'sublabels' in parent_data:
            for sublabel, sublabel_data in parent_data['sublabels'].items():
                # Sublabels have the top-level label as parent
                parent_map[sublabel] = parent_label
                
                # Set style for sublabel
                if 'color' in sublabel_data:
                    bg_color = hex_to_rgb(sublabel_data['color'])
                    text_color = get_text_color(sublabel_data['color'])
                    style_map[sublabel] = f"background-color: {bg_color}; color: {text_color};"
    
    def process_auto_label(match):
        """Process each auto_label opening tag match."""
        full_tag = match.group(0)
        
        # Extract labelname
        labelname_match = re.search(r'labelname="([^"]*)"', full_tag)
        if not labelname_match:
            return full_tag  # No labelname found, return unchanged
        
        labelname = labelname_match.group(1)
        
        # Determine parent attribute from label scheme
        parent_value = parent_map.get(labelname, "")
        parent_attr = f'parent="{parent_value}"'
        
        # Get style attribute from label scheme
        style = style_map.get(labelname, '')
        if style:
            style_attr = f'style="{style}"'
        else:
            style_attr = ''
        
        # Check if parent or style already exist
        has_parent = 'parent=' in full_tag
        has_style = 'style=' in full_tag
        
        # Build new tag
        # Remove existing parent/style if present
        if has_parent:
            full_tag = re.sub(r'\s*parent="[^"]*"', '', full_tag)
        if has_style:
            full_tag = re.sub(r'\s*style="[^"]*"', '', full_tag)
        
        # Insert new attributes before the closing >
        new_tag = full_tag[:-1]  # Remove closing >
        new_tag += f' {parent_attr} {style_attr}>'
        
        return new_tag
    
    # Pattern to match auto_label opening tags
    pattern = r'<auto_label[^>]*>'
    
    # Process all auto_label opening tags
    modified_html = re.sub(pattern, process_auto_label, html_content, flags=re.IGNORECASE)
    
    return modified_html



def compare_html_allow_auto_labels(merged_html: str, original_html: str) -> bool:
    """
    Compare two HTML strings character-by-character, considering them equivalent
    if the only differences are the presence or placement of <auto_label ...>
    and </auto_label> tags.

    Returns True when equal after stripping auto_label tags, else prints
    a concise diff context and returns False.
    """
    a = strip_auto_labels(merged_html)
    b = strip_auto_labels(original_html)
    if a == b:
        print("   ✓ HTMLs match when ignoring auto_label tags")
        return True
    # Find first index of difference
    min_len = min(len(a), len(b))
    diff_idx = None
    for i in range(min_len):
        if a[i] != b[i]:
            diff_idx = i
            print(f"   ✗ Difference at index {diff_idx}: '{a[i]}' vs '{b[i]}'")
            break
    if diff_idx is None and len(a) != len(b):
        diff_idx = min_len
    # Print a small window around the difference
    if diff_idx is not None:
        start = max(0, diff_idx - 50)
        end_a = min(len(a), diff_idx + 50)
        end_b = min(len(b), diff_idx + 50)
        print("   ✗ Difference found (ignoring auto_label):")
        print("--- merged_html (stripped) ---")
        print(a[start:end_a])
        print("--- original_html (stripped) ---")
        print(b[start:end_b])
    else:
        print("   ✗ Difference detected but could not locate index")
    return False