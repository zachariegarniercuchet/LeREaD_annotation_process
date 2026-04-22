from bs4 import BeautifulSoup


def extract_parent_level_annotations(html_content):
    """
    Extract all parent-level manual_label and auto_label annotations (where parent="").
    Returns a dict with keys: 'decision', 'legislation', 'secondary sources'
    Each value is a list of annotation dictionaries containing:
      - full_html: the complete annotation HTML
      - docid: document identifier
      - uri: resource URI
      - text_content: extracted text (no HTML tags)
      - sublabels: list of sublabel types found
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all manual_label and auto_label tags with parent=""
    parent_labels = soup.find_all(['manual_label', 'auto_label'], attrs={'parent': ''})
    
    annotations = {
        'decision': [],
        'legislation': [],
        'secondary sources': []
    }
    
    for label in parent_labels:
        labelname = label.get('labelname', '')
        
        if labelname not in annotations:
            continue
        
        # Extract sublabels
        sublabels = []
        for sublabel in label.find_all(['manual_label', 'auto_label'], recursive=False):
            sublabel_name = sublabel.get('labelname', '')
            sublabels.append(sublabel_name)
        
        # Recursively get all sublabels (nested)
        all_sublabels = [sl.get('labelname', '') for sl in label.find_all(['manual_label', 'auto_label'])]
        
        annotation_data = {
            'full_html': str(label),
            'docid': label.get('docid', ''),
            'uri': label.get('uri', ''),
            'text_content': label.get_text(strip=True),
            'direct_sublabels': sublabels,
            'all_sublabels': all_sublabels
        }
        
        annotations[labelname].append(annotation_data)
    
    return annotations



def get_sublabel_strings(annotations_dict, parent_label, sublabel_key, max_items=None):
    """
    Collect and display all strings for a given sublabel under a given parent label.

    annotations_dict: dict like `all_annotations` or `fewshot_annotations`
        with keys: 'decision', 'legislation', 'secondary sources'.
    parent_label: e.g. 'decision', 'legislation', 'secondary sources'.
    sublabel_key: e.g. 'title', 'fragment', 'citation', 'source', 'authors'.
    max_items: optional limit on how many strings to print/return (None = all).
    """
    from bs4 import BeautifulSoup
    
    if parent_label not in annotations_dict:
        print(f"Unknown parent label: {parent_label!r}. Available: {list(annotations_dict.keys())}")
        return []
    
    results = []
    annotations = annotations_dict[parent_label]
    
    for ann in annotations:
        html = ann.get('full_html', '')
        if not html:
            continue
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Distinguish between manual HTML annotations and JSON-style XML output
        if 'manual_label' in html or 'auto_label' in html:
            # Manual annotation: sublabels are <manual_label labelname="...">
            elements = soup.find_all(['manual_label', 'auto_label'], attrs={'labelname': sublabel_key})
        else:
            # JSON output format: sublabels are simple tags like <title>, <fragment>, etc.
            elements = soup.find_all(sublabel_key)
        
        for el in elements:
            text = el.get_text(strip=True)
            if text:
                results.append(text)
                if max_items is not None and len(results) >= max_items:
                    break
        if max_items is not None and len(results) >= max_items:
            break
    
    #print(f"Found {len(results)} strings for sublabel {sublabel_key!r} under parent label {parent_label!r}.")
    #for i, txt in enumerate(results, 1):
    #    print(f"[{i}] {txt}")
    
    return results



def get_ngrams(seq, n):
    return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]