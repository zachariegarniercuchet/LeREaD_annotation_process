from bs4 import BeautifulSoup
import itertools
import re
    
def extract_body(html_content: str) -> str:
    """
    Extract only the body content from HTML, excluding style, script, and head tags.
    Returns the exact string representation of the <body> element to keep reversibility.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    body = soup.find('body')
    if body is not None:
        return str(body)
    print("   ⚠ Warning: No <body> tag found, returning original content")
    return html_content

def remove_bookmarks(tokens: list) -> list:
    """Remove all 
      - <htmllabelizer_bookmark ...> tokens 
      - </htmllabelizer_bookmark ...> closing tokens 
      - tokens in between 
    from the token list."""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith('<htmllabelizer_bookmark'):
            # Find closing tag
            j = i + 1
            while j < len(tokens):
                if tokens[j].startswith('</htmllabelizer_bookmark'):
                    break
                j += 1
            # Remove from i to j (inclusive)
            del tokens[i:j+1]
            continue
        i += 1

    return [tok for tok in tokens if not tok.startswith('<htmllabelizer_bookmark')]

def is_tag_token(tok: str) -> bool:
    """Return True if token looks like an HTML tag (e.g., <...>)."""
    return len(tok) >= 3 and tok[0] == '<' and tok[-1] == '>'


def is_auto_label_tag(tok: str) -> bool:
    """Return 1 if token is an opening, 2 if it is a closing auto_label tag, 0 otherwise"""
    if not is_tag_token(tok):
        return 0
    # Accept variations with attributes on opening tag
    if tok.lower().startswith('<auto_label'):
        return 1
    if tok.lower().startswith('</auto_label'):
        return 2
    return 0

def is_manual_label_tag(tok):
    """Return 1 if token is an opening, 2 if it is a closing manual_label tag, 0 otherwise"""
    if not is_tag_token(tok):
        return 0
    # Accept variations with attributes on opening tag
    if tok.lower().startswith('<manual_label'):
        return 1
    if tok.lower().startswith('</manual_label'):
        return 2
    return 0

def strip_auto_labels(html: str) -> str:
    """Remove all <auto_label ...> and </auto_label> tags from html."""
    # Remove opening tags with any attributes
    html_no_open = re.sub(r"<\s*auto_label\b[^>]*>", "", html, flags=re.IGNORECASE)
    # Remove closing tags
    html_no_tags = re.sub(r"<\s*/\s*auto_label\s*>", "", html_no_open, flags=re.IGNORECASE)
    return html_no_tags





