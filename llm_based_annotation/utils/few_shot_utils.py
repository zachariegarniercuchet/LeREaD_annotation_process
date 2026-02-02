import re
from utils.html_cleaner import clean_tokens
from utils.tokenizer_utils import decode

def clean_token_manual_label(token, auto_transformation=False):
    """
    Clean and simplify manual_label HTML tokens by extracting only relevant attributes.
    
    This function processes HTML tokens containing manual_label tags and simplifies them
    by keeping only the attributes that are relevant for each specific label type. It can
    also optionally transform manual_label tags to auto_label tags.
    
    Args:
        token (str): An HTML token string, potentially containing a manual_label tag.
        auto_transformation (bool, optional): If True, converts manual_label tags to 
            auto_label tags. Defaults to False.
    
    Returns:
        str: The cleaned/simplified token with only relevant attributes, or the original
            token if it's not a manual_label tag.
    
    Behavior by labelname:
        - 'mention': Keeps labelname and docid attributes
        - 'title': Keeps labelname and titletype attributes
        - 'fragment': Keeps labelname, fragmentid, and fragmenttype attributes
        - 'reference': Keeps only labelname (no other attributes)
    
    Examples:
        >>> token = '<manual_label labelname="mention" docid="123" extra="ignore">'
        >>> clean_token_manual_label(token)
        '<manual_label labelname="mention" docid="123">'
        
        >>> token = '<manual_label labelname="title" titletype="main" extra="ignore">'
        >>> clean_token_manual_label(token, auto_transformation=True)
        '<auto_label labelname="title" titletype="main">'
        
        >>> token = '</manual_label>'
        >>> clean_token_manual_label(token, auto_transformation=True)
        '</auto_label>'
    """

    if token.startswith('<manual_label') and not token.startswith('</manual_label'):
        # Extract only relevant attributes based on labelname
        labelname_match = re.search(r'labelname="([^"]*)"', token)
        
        if labelname_match:
            labelname = labelname_match.group(1)

            if auto_transformation:
                # Change tag to auto_label
                simplified_tag = '<auto_label'
            else:
                # Keep tag as manual_label
                simplified_tag = '<manual_label'
            
            # Always keep labelname
            simplified_tag += f' labelname="{labelname}"'
            
            # Keep specific attributes based on labelname
            if labelname == 'mention':
                # For mention: keep docid and doctype
                docid_match = re.search(r'docid="([^"]*)"', token)
                
                if docid_match:
                    simplified_tag += f' docid="{docid_match.group(1)}"'
            
            elif labelname == 'title':
                # For title: keep titletype
                titletype_match = re.search(r'titletype="([^"]*)"', token)
                if titletype_match:
                    simplified_tag += f' titletype="{titletype_match.group(1)}"'
            
            elif labelname == 'fragment':
                # For fragment: keep fragmentid and fragmenttype
                fragmentid_match = re.search(r'fragmentid="([^"]*)"', token)
                fragmenttype_match = re.search(r'fragmenttype="([^"]*)"', token)
                
                if fragmentid_match:
                    simplified_tag += f' fragmentid="{fragmentid_match.group(1)}"'
                if fragmenttype_match:
                    simplified_tag += f' fragmenttype="{fragmenttype_match.group(1)}"'
            
            # For reference: only keep labelname (no other attributes)
            
            simplified_tag += '>'
            return simplified_tag
        else:
            # If no labelname found, keep original token
            return token
    elif token.startswith('</manual_label') and auto_transformation:
        # Replace closing tag with auto_label closing tag
        return '</auto_label>'
    else:
        # Not a manual_label tag, keep as-is
        return token
    


def extract_few_shot_examples(token_chunks):
    """
    Extract few-shot examples from the chunks.
    
    Args:
        token_chunks: List of token chunks to process
    
    Returns:
        list: List of tuples (input_tokens, output_tokens)
              - input_tokens: cleaned tokens without manual_label tags
              - output_tokens: tokens with simplified manual_label tags (only keeping relevant attributes)
    """
    examples = []
    
    for chunk in token_chunks:
        # Input: tokens cleaned from all manual_label tags
        input_chunk = clean_tokens(chunk, normalize=True, keep_manual_label=False, keep_bookmarks=False)
        
        # Output: tokens with simplified manual_label tags
        # Remove irrelevant attributes like style, parent, url from manual_label tags
        output_chunk = []
        
        for token in chunk:
            # Check if token is a manual_label opening tag
            cleaned_token = clean_token_manual_label(token, auto_transformation=True)

            output_chunk.append(cleaned_token)
        
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