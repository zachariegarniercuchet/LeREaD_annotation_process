# Resolution Utils

Utilities for extracting spans and clustering coresolutions in legal document annotations.

## Overview

This module provides tools to:
1. **Extract spans** from annotated HTML files with surrounding context
2. **Filter spans** to specific parent categories (legislation, decision, secondary sources)
3. **Cluster spans** based on text similarity to identify coresolutions

**Note:** This module imports core span extraction logic directly from `annotation_utils/IAA.py` to avoid code duplication. The resolution utilities extend the base functionality with clustering-specific features.

## Components

### `span_extraction.py`

Extends IAA.py span extraction with resolution-specific methods.

**Key Classes:**
- `Span`: Extends IAA.Span with `get_top_level_parent()` and `is_parent_category()` methods

**Key Functions:**
- `extract_spans_from_html(html_file, context_chars=200)`: Extract spans (uses IAA.py internally)
- `filter_parent_spans(spans, parent_categories)`: Filter to specific parent categories
- `get_context_around_element(element, plain_text, context_chars)`: Imported from IAA.py

### `clustering.py`

Creates coresolution clusters based on text similarity.

**Similarity Methods:**
- `lcs` - Longest Common Subsequence (flexible, allows gaps) **[RECOMMENDED]**
- `substring` - Longest Common Substring (stricter, no gaps)
- `ratio` - SequenceMatcher ratio (order-sensitive)
- `token` - Token-based Jaccard similarity

**Key Functions:**
- `create_coresolution_clusters(spans, similarity_threshold, similarity_method, group_by_parent)`: Create clusters
- `similarity_score(text1, text2, method)`: Calculate similarity between two texts
- `print_clusters(clusters)`: Pretty print clusters
- `analyze_cluster_statistics(clusters)`: Get cluster statistics

## Usage Example

```python
from utils import (
    extract_spans_from_html,
    filter_parent_spans,
    create_coresolution_clusters,
    analyze_cluster_statistics
)

# 1. Extract spans from HTML
spans = extract_spans_from_html('document.html', context_chars=200)

# 2. Filter to parent categories only
parent_spans = filter_parent_spans(
    spans,
    parent_categories=['legislation', 'decision', 'secondary sources']
)

# 3. Create clusters based on similarity
clusters = create_coresolution_clusters(
    parent_spans,
    similarity_threshold=0.6,      # 0-1: higher = stricter
    similarity_method='lcs',        # 'lcs', 'substring', 'ratio', 'token'
    group_by_parent=True           # Cluster within same parent category
)

# 4. Analyze results
stats = analyze_cluster_statistics(clusters)
print(f"Total clusters: {stats['total_clusters']}")
print(f"Average cluster size: {stats['avg_cluster_size']:.2f}")
```

## Clustering Parameters

### Similarity Threshold
Controls how similar texts must be to cluster together:
- `0.4-0.5`: Very lenient, groups loosely related texts
- `0.6`: **Recommended** - balances precision and recall
- `0.7-0.8`: Strict, only very similar texts cluster
- `0.9-1.0`: Very strict, nearly identical texts only

### Similarity Methods

| Method | Best For | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| `lcs` | **General use** | Flexible, allows gaps in text | May overcluster |
| `substring` | Exact matches | Finds continuous matching text | Misses paraphrases |
| `ratio` | Order-sensitive | Considers word order | Sensitive to rearrangement |
| `token` | Keywords | Good for shared terminology | Ignores order/context |

### Group By Parent
- `True`: **Recommended** - Only clusters spans with same parent category (legislation/decision/secondary sources)
- `False`: Clusters across all parent categories

## Output Format

### Span Object
```python
Span(
    text="Supreme Court of Canada",
    start=1234,
    end=1260,
    labelname="decision",
    attributes={'docid': '2019SCC65', 'uri': '...'},
    context_text="...surrounding text...",
    parent="decision"
)
```

### Clusters Dictionary
```python
{
    0: [Span1, Span2, Span3],  # Cluster 0: 3 similar spans
    1: [Span4, Span5],          # Cluster 1: 2 similar spans
    2: [Span6]                  # Cluster 2: 1 singleton span
}
```

## Integration with IAA.py

**This module directly imports from `annotation_utils/IAA.py` to avoid code duplication:**
- Core span extraction logic comes from IAA.py (single source of truth)
- The `Span` class extends IAA.Span with resolution-specific helper methods
- Same context-based extraction ensures consistency across IAA evaluation and resolution clustering
- Changes to IAA.py span extraction automatically apply to resolution utilities

## Tips

1. **Start with recommended settings**: `similarity_threshold=0.6`, `method='lcs'`, `group_by_parent=True`
2. **Iterate on threshold**: If too many clusters → lower threshold; if clusters too loose → raise threshold
3. **Inspect multi-span clusters**: Focus on clusters with 2+ spans for coresolution analysis
4. **Export for downstream tasks**: Use JSON export for further processing/annotation
5. **Validate with IAA**: Cross-check clustering with inter-annotator agreement metrics

## Dependencies

- `beautifulsoup4`: HTML parsing
- `difflib`: Text similarity (Python stdlib)
- `collections`: Data structures (Python stdlib)
