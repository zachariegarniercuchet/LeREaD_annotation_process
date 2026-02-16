"""
Span extraction utilities - imports directly from IAA.py to avoid redundancy.
Adds resolution-specific helper methods.
"""

import sys
import os

# Add annotation_utils to path to import from IAA.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'annotation_utils'))

# Import core functions directly from IAA.py (single source of truth)
from IAA import Span as BaseSpan, extract_spans_from_html, get_context_around_element

from typing import List


# Extend Span class with resolution-specific methods
class Span(BaseSpan):
    """Extends IAA.Span with resolution-specific helper methods."""
    
    def __init__(self, text: str, start: int, end: int, labelname: str, 
                 attributes: dict, context_text: str = ""):
        super().__init__(text, start, end, labelname, attributes, context_text)
        self.parent = attributes.get('parent', '')
    
    def get_top_level_parent(self) -> str:
        """
        Extract the top-level parent from comma-separated parent attribute.
        Returns one of: 'legislation', 'decision', 'secondary sources', or empty string.
        """
        if not self.parent:
            return ''
        
        # Get first element in comma-separated parent chain
        top_parent = self.parent.split(',')[0].strip().lower()
        
        # Normalize to standard names
        if top_parent in ['legislation', 'decision', 'secondary sources']:
            return top_parent
        
        return ''
    
    def is_parent_category(self) -> bool:
        """Check if this span is one of the main parent categories."""
        return self.labelname.lower() in ['legislation', 'decision', 'secondary sources']


# Wrapper to return extended Span objects
def extract_spans_from_html_extended(html_file: str, label_type=["manual_label"], context_chars: int = 200) -> List[Span]:
    """
    Extract spans using IAA.extract_spans_from_html, returning extended Span objects.
    
    Args:
        html_file: Path to the HTML file
        label_type: List of label types to find (default: ["manual_label"])
        context_chars: Number of characters before/after for context (default: 200)
        
    Returns:
        List of extended Span objects with resolution-specific methods
    """
    from bs4 import BeautifulSoup
    
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    body = soup.find('body')
    
    if not body:
        return []
    
    # Get plain text from body (all HTML stripped)
    plain_text = body.get_text()
    
    # Find all manual_label or auto_label elements
    labels = body.find_all(label_type)
    
    spans = []
    
    for label in labels:
        labelname = label.get('labelname', '')
        
        # Get all attributes except style, labelname
        attributes = {k: v for k, v in label.attrs.items() 
                     if k not in ['style', 'labelname']}
        
        # Get the label's text content (normalized)
        text = label.get_text()
        normalized_text = ' '.join(text.split())
        
        if not normalized_text:
            continue
        
        # Extract context around this label - THIS IS THE KEY INNOVATION
        context_text, start_idx, end_idx = get_context_around_element(label, plain_text, context_chars)
        
        # Use extended Span class
        span = Span(normalized_text, start_idx, end_idx, labelname, attributes, context_text)
        spans.append(span)
    
    return spans


def filter_parent_spans(spans: List[Span], parent_categories: List[str] = None) -> List[Span]:
    """
    Filter spans to only include those with specified parent categories.
    
    Args:
        spans: List of all spans
        parent_categories: List of parent categories to filter by.
                          Default: ['legislation', 'decision', 'secondary sources']
    
    Returns:
        Filtered list of spans that have the specified parent categories
    """
    if parent_categories is None:
        parent_categories = ['legislation', 'decision', 'secondary sources']
    
    # Normalize category names
    normalized_categories = [cat.lower() for cat in parent_categories]
    
    filtered = []
    
    for span in spans:
        # Check if labelname is a parent category
        if span.labelname.lower() in normalized_categories:
            filtered.append(span)
            continue
        
        # Check if top-level parent matches
        top_parent = span.get_top_level_parent()
        if top_parent in normalized_categories:
            filtered.append(span)
    
    return filtered


# Alias for convenience - this is the main function to use
extract_spans_from_html = extract_spans_from_html_extended