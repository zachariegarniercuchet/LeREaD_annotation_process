from bs4 import BeautifulSoup


# ========================= BUG OF MANUAL LABEL CLASS =========================

def clean_manual_label_classes(html_content):
    """
    Remove class attributes from all <manual_label> elements in the HTML.
    
    Args:
        html_content (str): The HTML content as a string
        
    Returns:
        str: The cleaned HTML content
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all manual_label elements
    manual_labels = soup.find_all('manual_label')
    
    # Remove class attribute from each manual_label element
    for label in manual_labels:
        if 'class' in label.attrs:
            del label.attrs['class']
    
    return str(soup)

# ==================================================================================




# ======================= BUG OF REFERENCE + SOME ATTRIBUTES =======================

def clean_label_attributes(html_content):
    """
    Remove unwanted attributes from <manual_label> elements where
    labelname is in {'reference', 'source', 'authors'}.

    Only keeps: labelname, parent, and style attributes.
    
    Args:
        html_content (str): The HTML content as a string
        
    Returns:
        str: The cleaned HTML content
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Label names we want to clean
    target_labels = {'reference', 'source', 'authors'}
    
    # Allowed attributes
    allowed_attrs = {'labelname', 'parent', 'style'}
    
    # Find all manual_label elements
    for label in soup.find_all('manual_label'):
        labelname = label.attrs.get('labelname')
        
        if labelname in target_labels:
            attrs_to_remove = [
                attr for attr in label.attrs
                if attr not in allowed_attrs
            ]
            
            for attr in attrs_to_remove:
                del label.attrs[attr]
    
    return str(soup)

# ==================================================================================



# ======================= CLEAN GLOBAL STYLE POLLUTION =======================

def clean_global_style_pollution(html_content):
    """
    Removes unwanted global CSS rules from all tag styles.
    If a style attribute contains only the unwanted rules, it is removed entirely.
    Otherwise, only the unwanted rules are removed.
    
    Args:
        html_content (str): The HTML content as a string
        
    Returns:
        str: The cleaned HTML content
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    UNWANTED_PROPERTIES = {
        'max-width': '100%',
        'box-sizing': 'border-box',
        'overflow-wrap': 'break-word',
        'word-break': 'break-word'
    }

    for tag in soup.find_all(True):  # all tags
        style = tag.get('style')
        if not style:
            continue

        cleaned_rules = []

        for rule in style.split(';'):
            rule = rule.strip()
            if not rule or ':' not in rule:
                continue

            prop, value = map(str.strip, rule.split(':', 1))

            # Skip unwanted exact matches
            if prop in UNWANTED_PROPERTIES and value == UNWANTED_PROPERTIES[prop]:
                continue

            cleaned_rules.append(f"{prop}: {value}")

        if cleaned_rules:
            tag['style'] = '; '.join(cleaned_rules) + ';'
        else:
            del tag.attrs['style']

    return str(soup)

# ============================================================================



# ======================= CHANGE REFERENCE TO CITATION =======================

def reference_to_citation(html_content):
    """
    Convert <manual_label> elements with labelname 'reference' to 'citation'.
    Also updates the JSON label tree in the HTML comment to change 'reference' to 'citation'.
    
    Args:
        html_content (str): The HTML content as a string
        
    Returns:
        str: The modified HTML content
    """
    import json
    import re
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Update labelname attributes in manual_label elements
    for label in soup.find_all('manual_label'):
        if label.attrs.get('labelname') == 'reference':
            label.attrs['labelname'] = 'citation'

    # Update labelname attributes in auto_label elements
    for label in soup.find_all('auto_label'):
        if label.attrs.get('labelname') == 'reference':
            label.attrs['labelname'] = 'citation'
    
    # Update the JSON label tree in the HTML comment
    html_str = str(soup)
    
    # Find the HTMLLabelizer comment with JSON
    comment_pattern = r'(<!-- HTMLLabelizer\s*)(\{.*?\})(\s*-->)'
    match = re.search(comment_pattern, html_str, re.DOTALL)
    
    if match:
        json_str = match.group(2)
        try:
            # Parse the JSON
            label_tree = json.loads(json_str)
            
            # Recursively rename 'reference' keys to 'citation' in the label tree
            def rename_reference_keys(obj):
                if isinstance(obj, dict):
                    new_dict = {}
                    for key, value in obj.items():
                        # If the key is 'reference', rename it to 'citation'
                        new_key = 'citation' if key == 'reference' else key
                        # Recursively process the value
                        new_dict[new_key] = rename_reference_keys(value)
                    return new_dict
                elif isinstance(obj, list):
                    return [rename_reference_keys(item) for item in obj]
                else:
                    return obj
            
            # Apply the renaming
            label_tree = rename_reference_keys(label_tree)
            
            # Convert back to JSON string with proper formatting
            new_json_str = json.dumps(label_tree, indent=2)
            
            # Replace the old JSON with the new one in the HTML
            html_str = html_str[:match.start()] + match.group(1) + new_json_str + match.group(3) + html_str[match.end():]
            
        except json.JSONDecodeError:
            # If JSON parsing fails, continue without modifying the comment
            pass
    
    return html_str

def add_verified_attribute(html_content):
    """
    Add a 'verified' attribute with value 'true' to all <manual_label> elements.
    
    Args:
        html_content (str): The HTML content as a string
        
    Returns:
        str: The modified HTML content
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for label in soup.find_all('manual_label'):
        label.attrs['verified'] = 'False'

    for label in soup.find_all('auto_label'):
        label.attrs['verified'] = 'False'
    
    return str(soup)

if __name__ == "__main__":
    import os
    
    # File paths
    annotated_dict = {
        "GL": ["2001BCSC1342_annotated_GL"],
        "EG": ["1999CanLII7320_annotated_EG", "2019SCC65_annotated_EG", "2016QCCS1184_annotated_EG"],
        "VP": ["1999CanLII7320_annotated_VP", "2019SCC65_annotated_VP", "2001BCSC1342_annotated_VP"],
        'llm': ["2019SCC65\\2019SCC65_llm_v1.0", "2019SCC65\\2019SCC65_llm_v1.1", "2019SCC65\\2019SCC65_llm_v1.2", "2019SCC65\\2019SCC65_llm_v1.3"]}
    #anno = "VP"  # Options: "GL", "EG", "VP"
    #file_name = f"2019SCC65_annotated_{anno}" #"2016QCCS1184_annotated_{anno}" #"1999CanLII7320_annotated_{anno}" #"2001BCSC1342_annotated_{anno}" # 2019SCC65_annotated_{anno}

    for anno in annotated_dict.keys():
        for file_name in annotated_dict[anno]:
            input_file = rf"data\Documents_Annotés\{anno}\{file_name}.html"
            output_file = rf"data\Documents_Annotés\{anno}\{file_name}_v1.html"
    
            # Read the HTML file
            with open(input_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Clean the manual_label class attributes
            cleaned_html = clean_manual_label_classes(html_content)
            
            # Clean reference attributes (remove unwanted attributes like titletype, docid, uri, etc.)
            cleaned_html = clean_label_attributes(cleaned_html)

            # CLEAN GLOBAL STYLE POLLUTION
            cleaned_html = clean_global_style_pollution(cleaned_html)

            cleaned_html = reference_to_citation(cleaned_html)

            cleaned_html = add_verified_attribute(cleaned_html)
            
            # Save the result
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_html)
            
            print(f"Cleaned HTML saved to: {output_file}")
