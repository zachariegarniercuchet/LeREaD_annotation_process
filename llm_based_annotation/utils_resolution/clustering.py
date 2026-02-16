"""
Coresolution clustering based on text similarity.
Groups spans that likely refer to the same legal reference.
"""

from typing import List, Dict, Tuple
from collections import defaultdict
import difflib


def longest_common_substring(s1: str, s2: str) -> str:
    """
    Find the longest common substring between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        The longest common substring
    """
    # Use SequenceMatcher to find matching blocks
    matcher = difflib.SequenceMatcher(None, s1, s2)
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    
    if match.size == 0:
        return ""
    
    return s1[match.a:match.a + match.size]


def longest_common_subsequence_length(s1: str, s2: str) -> int:
    """
    Calculate the length of the longest common subsequence (LCS).
    More flexible than substring as it allows gaps.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Length of the LCS
    """
    m, n = len(s1), len(s2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def similarity_score(text1: str, text2: str, method: str = 'lcs') -> float:
    """
    Calculate similarity score between two text strings.
    
    Args:
        text1: First text string
        text2: Second text string
        method: Similarity method to use:
                - 'lcs': Longest Common Subsequence ratio (default)
                - 'substring': Longest Common Substring ratio
                - 'ratio': SequenceMatcher ratio (considers order)
                - 'token': Token-based Jaccard similarity
        
    Returns:
        Similarity score between 0 and 1
    """
    # Normalize texts: lowercase and strip whitespace
    t1 = ' '.join(text1.lower().split())
    t2 = ' '.join(text2.lower().split())
    
    if not t1 or not t2:
        return 0.0
    
    if t1 == t2:
        return 1.0
    
    if method == 'substring':
        # Longest common substring ratio
        lcs = longest_common_substring(t1, t2)
        return len(lcs) / min(len(t1), len(t2))
    
    elif method == 'lcs':
        # Longest common subsequence ratio (more flexible)
        lcs_len = longest_common_subsequence_length(t1, t2)
        return lcs_len / max(len(t1), len(t2))
    
    elif method == 'ratio':
        # SequenceMatcher ratio (considers order)
        return difflib.SequenceMatcher(None, t1, t2).ratio()
    
    elif method == 'token':
        # Token-based Jaccard similarity
        tokens1 = set(t1.split())
        tokens2 = set(t2.split())
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def create_coresolution_clusters(spans: List, 
                                 similarity_threshold: float = 0.6,
                                 similarity_method: str = 'lcs',
                                 group_by_parent: bool = True) -> Dict[int, List]:
    """
    Create clusters of spans that likely refer to the same legal reference.
    Uses text similarity to group spans together.
    
    Args:
        spans: List of Span objects
        similarity_threshold: Minimum similarity score to cluster together (0-1)
        similarity_method: Method for calculating similarity ('lcs', 'substring', 'ratio', 'token')
        group_by_parent: If True, only cluster spans with the same parent category
        
    Returns:
        Dictionary mapping cluster_id to list of spans
    """
    if not spans:
        return {}
    
    # Group spans by parent category if requested
    if group_by_parent:
        parent_groups = defaultdict(list)
        for span in spans:
            # Get parent category
            if hasattr(span, 'get_top_level_parent'):
                parent = span.get_top_level_parent()
            else:
                parent = span.labelname.lower() if span.labelname.lower() in ['legislation', 'decision', 'secondary sources'] else 'other'
            
            parent_groups[parent].append(span)
    else:
        parent_groups = {'all': spans}
    
    # Cluster within each parent group
    all_clusters = {}
    cluster_id = 0
    
    for parent_cat, parent_spans in parent_groups.items():
        if not parent_spans:
            continue
        
        # Track which spans have been clustered
        clustered = set()
        
        for i, span1 in enumerate(parent_spans):
            if i in clustered:
                continue
            
            # Start a new cluster with this span
            cluster = [span1]
            clustered.add(i)
            
            # Find all similar spans
            for j, span2 in enumerate(parent_spans):
                if j <= i or j in clustered:
                    continue
                
                # Calculate similarity
                sim = similarity_score(span1.text, span2.text, method=similarity_method)
                
                if sim >= similarity_threshold:
                    cluster.append(span2)
                    clustered.add(j)
            
            # Add cluster if it has at least one span
            if cluster:
                all_clusters[cluster_id] = cluster
                cluster_id += 1
    
    return all_clusters


def print_clusters(clusters: Dict[int, List], max_text_len: int = 50):
    """
    Pretty print the clusters for inspection.
    
    Args:
        clusters: Dictionary mapping cluster_id to list of spans
        max_text_len: Maximum length of text to display
    """
    print(f"\n{'='*80}")
    print(f"CORESOLUTION CLUSTERS")
    print(f"{'='*80}")
    print(f"\nTotal clusters: {len(clusters)}")
    
    for cluster_id, spans in sorted(clusters.items()):
        print(f"\n{'─'*80}")
        print(f"Cluster {cluster_id} ({len(spans)} spans)")
        print(f"{'─'*80}")
        
        for idx, span in enumerate(spans, 1):
            text_display = span.text if len(span.text) <= max_text_len else span.text[:max_text_len] + "..."
            parent_info = ""
            
            if hasattr(span, 'get_top_level_parent'):
                parent = span.get_top_level_parent()
                if parent:
                    parent_info = f" [Parent: {parent}]"
            
            print(f"  {idx}. {text_display}")
            print(f"     Label: {span.labelname}{parent_info}")
            
            # Show key attributes
            if span.attributes:
                attrs_display = []
                for key in ['docid', 'uri', 'titletype', 'fragmentid']:
                    if key in span.attributes:
                        value = span.attributes[key]
                        if len(value) > 40:
                            value = value[:40] + "..."
                        attrs_display.append(f"{key}={value}")
                
                if attrs_display:
                    print(f"     Attributes: {', '.join(attrs_display)}")
    
    print(f"\n{'='*80}\n")


def print_clusters_simple(clusters: Dict[int, List], max_text_len: int = 60):
    """
    Simple print of clusters - just cluster ID and list of span texts.
    
    Args:
        clusters: Dictionary mapping cluster_id to list of spans
        max_text_len: Maximum length of text to display per span
    """
    print(f"\nCoresolution Clusters: {len(clusters)} total\n")
    
    for cluster_id, spans in sorted(clusters.items()):
        print(f"Cluster {cluster_id}: [{len(spans)} spans]")
        for span in spans:
            text_display = span.text if len(span.text) <= max_text_len else span.text[:max_text_len] + "..."
            print(f"  - {text_display}")
        print()


def analyze_cluster_statistics(clusters: Dict[int, List]) -> Dict:
    """
    Calculate statistics about the clusters.
    
    Args:
        clusters: Dictionary mapping cluster_id to list of spans
        
    Returns:
        Dictionary with statistics
    """
    if not clusters:
        return {
            'total_clusters': 0,
            'total_spans': 0,
            'avg_cluster_size': 0,
            'max_cluster_size': 0,
            'singleton_clusters': 0
        }
    
    cluster_sizes = [len(spans) for spans in clusters.values()]
    total_spans = sum(cluster_sizes)
    singleton_clusters = sum(1 for size in cluster_sizes if size == 1)
    
    stats = {
        'total_clusters': len(clusters),
        'total_spans': total_spans,
        'avg_cluster_size': total_spans / len(clusters),
        'max_cluster_size': max(cluster_sizes),
        'min_cluster_size': min(cluster_sizes),
        'singleton_clusters': singleton_clusters,
        'multi_span_clusters': len(clusters) - singleton_clusters
    }
    
    return stats
