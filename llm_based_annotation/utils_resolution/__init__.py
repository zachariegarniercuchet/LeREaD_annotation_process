"""
Resolution utilities for span extraction and coresolution clustering.

Uses IAA.py as the source of truth for span extraction to avoid code duplication.
"""

from utils.span_extraction import Span, extract_spans_from_html, filter_parent_spans, get_context_around_element
from utils.clustering import (
    create_coresolution_clusters, 
    longest_common_substring, 
    similarity_score,
    print_clusters,
    print_clusters_simple,
    analyze_cluster_statistics
)

__all__ = [
    'Span',
    'extract_spans_from_html',
    'filter_parent_spans',
    'get_context_around_element',
    'create_coresolution_clusters',
    'longest_common_substring',
    'similarity_score',
    'print_clusters',
    'print_clusters_simple',
    'analyze_cluster_statistics'
]
