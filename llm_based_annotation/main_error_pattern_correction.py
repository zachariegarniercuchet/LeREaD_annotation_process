
import os
import glob
import re

def end_punct_error_correction(html_content, debug=False):
    """
    Correction rule: auto_label tags should never end with comma.
    Move comma before </auto_label> to after the closing tag.
    If formatting tags (</i>, </b>) are between comma and </auto_label>, wrap the comma in those tags.
    Example: ,</i></auto_label> becomes </i></auto_label><i>,</i>
    """
    # Pattern to match comma, optional closing formatting tags, then </auto_label>
    # Group 1: comma
    # Group 2: any closing formatting tags (</i>, </b>, etc.)
    pattern = r'(,)((?:</[ib]>)*)(</auto_label>)'
    
    def replace_comma(match):
        comma = match.group(1)
        closing_tags = match.group(2)
        auto_label_close = match.group(3)
        
        if closing_tags:
            # Extract tag names from closing tags and create opening tags
            # </i></b> -> ['i', 'b']
            tag_names = re.findall(r'</([ib])>', closing_tags)
            # Create opening tags in same order: <i><b>
            opening_tags = ''.join(f'<{tag}>' for tag in tag_names)
            # Result: </i></b></auto_label><i><b>,</b></i>
            # But we need to reverse closing tags: <i><b>,</b></i>
            reversed_closing_tags = ''.join(f'</{tag}>' for tag in reversed(tag_names))
            return f'{closing_tags}{auto_label_close}{opening_tags}{comma}{reversed_closing_tags}'
        else:
            # No formatting tags, just move comma outside
            return f'{auto_label_close}{comma}'
    
    if debug:
        matches = list(re.finditer(pattern, html_content))
        if matches:
            print(f"\n=== END PUNCTUATION CORRECTION (COMMA) ===")
            print(f"Found {len(matches)} matches to correct:\n")
            for i, match in enumerate(matches, 1):
                # Get context: find previous <auto_label> tag
                start_pos = match.start()
                context_start = html_content.rfind('<auto_label', 0, start_pos)
                if context_start == -1:
                    context_start = max(0, start_pos - 100)  # Fallback: 100 chars before
                
                context_before = html_content[context_start:match.start()]
                matched_text = match.group(0)
                after_text = replace_comma(match)
                
                print(f"Match {i}/{len(matches)}:")
                print(f"  CONTEXT: ...{context_before}")
                print(f"  BEFORE:  {matched_text}")
                print(f"  AFTER:   {after_text}")
                input("  Press Enter to continue...")
        else:
            print("\n=== END PUNCTUATION CORRECTION (COMMA) ===")
            print("No matches found.")
    
    # Replace with closing tag first, then punctuation
    corrected_content = re.sub(pattern, replace_comma, html_content)
    
    return corrected_content

def quotation_error_correction(html_content, debug=False):
    """
    Correction rule: For <auto_label> tags with labelname="title" and parent="secondary sources",
    move quotation marks from inside to outside the tags.
    
    Example: <auto_label labelname="title" parent="secondary sources">"Text"</auto_label>
    Becomes: "<auto_label labelname="title" parent="secondary sources">Text</auto_label>"
    
    Handles both ASCII quotes (" ') and Unicode curly quotes (" " ' ')
    """
    # Pattern with explicit Unicode escapes for curly quotes:
    # \u0022 = " (ASCII double quote)
    # \u201C = " (left double quotation mark) 
    # \u201D = " (right double quotation mark)
    # \u0027 = ' (ASCII single quote)
    # \u2018 = ' (left single quotation mark)
    # \u2019 = ' (right single quotation mark)
    pattern = r'(<auto_label[^>]*>)([\u0022\u201C\u201D\u0027\u2018\u2019])(.+?)([\u0022\u201C\u201D\u0027\u2018\u2019])(</auto_label>)'
    
    def replace_quotes(match):
        opening_tag = match.group(1)
        opening_quote = match.group(2)
        content = match.group(3)
        closing_quote = match.group(4)
        closing_tag = match.group(5)
        
        # Only process if this has the required attributes
        if 'labelname="title"' not in opening_tag or 'parent="secondary sources"' not in opening_tag:
            # Not the target, return unchanged
            return match.group(0)
        
        # Move quotes outside: opening_quote + opening_tag + content + closing_tag + closing_quote
        return f'{opening_quote}{opening_tag}{content}{closing_tag}{closing_quote}'
    
    if debug:
        matches = list(re.finditer(pattern, html_content))
        # Filter to only show matches that will actually be changed
        relevant_matches = []
        for match in matches:
            opening_tag = match.group(1)
            if 'labelname="title"' in opening_tag and 'parent="secondary sources"' in opening_tag:
                relevant_matches.append(match)
        
        if relevant_matches:
            print(f"\n=== QUOTATION MARK CORRECTION (Title in Secondary Sources) ===")
            print(f"Found {len(relevant_matches)} matches to correct:\n")
            for i, match in enumerate(relevant_matches, 1):
                # Get context: find previous </auto_label> or <auto_label> tag
                start_pos = match.start()
                context_start = max(
                    html_content.rfind('</auto_label>', 0, start_pos),
                    html_content.rfind('<auto_label', 0, start_pos)
                )
                if context_start == -1:
                    context_start = max(0, start_pos - 100)  # Fallback: 100 chars before
                else:
                    # Move past the tag
                    context_start = html_content.find('>', context_start) + 1
                
                context_before = html_content[context_start:match.start()]
                matched_text = match.group(0)
                after_text = replace_quotes(match)
                
                print(f"Match {i}/{len(relevant_matches)}:")
                print(f"  CONTEXT: ...{context_before}")
                print(f"  BEFORE:  {matched_text}")
                print(f"  AFTER:   {after_text}")
                input("  Press Enter to continue...")
        else:
            print("\n=== QUOTATION MARK CORRECTION (Title in Secondary Sources) ===")
            print("No matches found.")
    
    # Apply the replacement
    corrected_content = re.sub(pattern, replace_quotes, html_content)
    
    return corrected_content

def main(debug=False):
    # Load all html finishing by v1.3.html in the specified directory
    base_dir = r"C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process\data\Documents_Annotés\llm\p2_c500_fsselected-30_mgpt-5.2"
    pattern = os.path.join(base_dir, "*_v1.3.html")
    
    # Get all matching files
    html_files = glob.glob(pattern)
    print(f"Found {len(html_files)} HTML files to process:")
    for file in html_files:
        print(f"  - {os.path.basename(file)}")
    
    # Process each file
    for file_num, html_file in enumerate(html_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {file_num}/{len(html_files)}: {os.path.basename(html_file)}")
        print(f"{'='*60}")
        
        # Read the HTML content
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Apply error correction functions
        html_content = end_punct_error_correction(html_content, debug=debug)
        html_content = quotation_error_correction(html_content, debug=debug)
        
        # Save the corrected HTML content back in the same location with v1.4
        output_file = html_file.replace("_v1.3.html", "_v1.4.html")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✓ Processed and saved: {os.path.basename(output_file)}")
    
    print(f"\n{'='*60}")
    print(f"Total files processed: {len(html_files)}")
  
    return html_files



if __name__ == "__main__":
    debug = True
    main(debug)