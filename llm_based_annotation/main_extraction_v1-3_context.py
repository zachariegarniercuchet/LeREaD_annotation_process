import os

from utils_extraction import extract_few_shot_examples_from_labels
from utils_extraction import SUBLABEL_DEFINITIONS_V1, SUBLABEL_DEFINITIONS_V2
from utils_extraction import extract_body, tokenize, clean_tokens, decode, is_auto_label_tag
from utils_extraction import chunk_tokens, flatten_token_chunks, merge_sentences_with_heuristics_tokens
from utils_extraction import extract_few_shot_examples
from utils_extraction import select_few_shot, prepare_label_tokens
from utils_extraction import merge_tokens_with_auto_labels, merge_tokens_general, add_attributes_to_auto_labels, compare_html_allow_auto_labels, correct_tokens_brackets, check_tokens_brackets
from models import GPTAssistant
from utils_extraction import process_labels
from utils_extraction import clean_html_formatting


import json
import spacy
import re

project_root = r"C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process"



def main_few_shot_selection(filename, sublabel_config, distribution, n_few_shot=30):
    

    # Load the selected few-shot examples JSON
    fs_json_path = fr"{project_root}\few_shot_selection_tool\second_selected\combined_v3_with_sources_manual_label.json"

    with open(fs_json_path, 'r', encoding='utf-8') as file:
        fs_data = json.load(file)
    print(f"   ✓ Loaded {len(fs_data)} examples from: {fs_json_path}")
    
    # Get all unique source files and print them
    source_files = sorted(list(set([item.get('source_file', 'unknown') for item in fs_data])))
    print(f"\n   Source files in few-shot collection:")
    for sf in source_files:
        count = sum(1 for item in fs_data if item.get('source_file') == sf)
        print(f"      - {sf} ({count} examples)")
    
    # Apply filters:
    # 1. Mask filter: Exclude examples from the same document being annotated (avoid data leakage)
    # 2. Manual filter: Only keep examples where "selected" == true
    current_doc_base = filename
    
    filtered_examples = []
    excluded_same_doc = 0
    excluded_not_selected = 0
    
    for item in fs_data:
        source_file = item.get('source_file', '')
        
        # Filter 1: Check if the current document name appears in the source file (MASK FILTER)
        if current_doc_base in source_file:
            excluded_same_doc += 1
            continue
        
        # Filter 2: Only keep examples with "selected" == true (MANUAL FILTER)
        if not item.get('selected', False):
            excluded_not_selected += 1
            continue
        
        # Extract input/output from the example
        if 'example' in item and 'input' in item['example'] and 'output' in item['example']:
            filtered_examples.append({
                'input': item['example']['input'],
                'output': item['example']['output'],
                'source_file': source_file
            })
    
    print(f"\n   ✓ Filtering results:")
    print(f"      - Excluded (same document): {excluded_same_doc}")
    print(f"      - Excluded (not selected): {excluded_not_selected}")
    print(f"      - Retained: {len(filtered_examples)}")
    
    # Show source files of retained examples
    retained_sources = {}
    for ex in filtered_examples:
        sf = ex['source_file']
        retained_sources[sf] = retained_sources.get(sf, 0) + 1
    
    print(f"\n   ✓ Retained examples come from:")
    for sf, count in sorted(retained_sources.items()):
        print(f"      - {sf}: {count} examples")
    
    # Select the required number of examples
    if len(filtered_examples) > n_few_shot:
        selected_examples_dicts = filtered_examples[:n_few_shot]
    else:
        selected_examples_dicts = filtered_examples
    
    
    
    print(f"\n   ✓ Simplifying outputs to parent-level extraction...")
    simplified_examples = []
    for ex in selected_examples_dicts:
        examples = extract_few_shot_examples_from_labels(tokenize(ex['output']), sublabel_config=sublabel_config)
        simplified_examples.extend(examples)
    
    # Convert to list of tuples (input, output)
    selected_few_shot_examples = select_few_shot(
        examples=simplified_examples, 
        n=n_few_shot,
        method="distributed",
        list_of_labels=sublabel_config["new_labels"],
        distribution=distribution
    )
    
    print(f"   ✓ Selected {len(selected_few_shot_examples)} few-shot examples for processing.")
    
    return selected_few_shot_examples[:n_few_shot]


def main_post_processing(processed_label, html_content, original_tokens):

    # Useless verification to check if the tokens are the same after processing (except for the auto labels)
    t1 = []
    t2 = []
    for token in processed_label:
        if not is_auto_label_tag(token) in [1, 2]:
            t1.append(token)


    for token in original_tokens:
        if not is_auto_label_tag(token) in [1, 2]:
            t2.append(token)

    assert t1 == t2, "The tokens are different after processing, which should not happen as we are only adding auto_label tags without changing the original tokens."


    processed_html = decode(processed_label)

    print(f"HTML length: {len(processed_html)}")

    # ---------- Add style and parent to auto_label tags ----------
    processed_html_content = add_attributes_to_auto_labels(processed_html)


    # Last check of the final processed_html_content with the original HTML, ignoring the auto_label tags which are not present in the original HTML but only in the processed one.
    comparison_result = compare_html_allow_auto_labels(processed_html_content, html_content)
    assert comparison_result, "The processed HTML content does not match the original HTML content when ignoring auto_label tags."

    return processed_html_content


    
def get_all_html_files_in(folder_path):
    """
    Returns:
        dict[str, str]: {filename_without_extension: html_content}
    """
    files = {}
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)

        if not os.path.isfile(full_path):
            continue

        name, ext = os.path.splitext(entry)
        if ext.lower() not in {".html", ".htm"}:
            continue

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                files[name] = f.read()
        except UnicodeDecodeError:
            with open(full_path, "r", encoding="latin-1") as f:
                files[name] = f.read()

    return files

def get_hyperparameters():
    # ---------- Define Hyperparameters ----------
    min_tokens = 500
    fs_min_tokens = 100
    fs_mode = "selected"  # "random" or "selected"
    model_name = "gpt-5.2"

    n_few_shot = 30  # Number of few-shot examples to use

    prompt_version = "2"
    with_context = 200  # Number of tokens to include as context
    

    # USE CONTEXT VERSION OF PROMPT
    prompt_path = fr"{project_root}\llm_based_annotation\utils_extraction\prompts\simplified_sublabels_extraction_from_parent.txt"

    sublabel_definitions = SUBLABEL_DEFINITIONS_V2
    


    return min_tokens, fs_min_tokens, fs_mode, model_name, n_few_shot, prompt_version, prompt_path, sublabel_definitions, with_context


def get_label_config(out_version):

    if out_version == "v1.1":
        sublabel_config = {
        "parent":["decision", "legislation", "secondary sources"], # only extract sublabels under these parents
        "already_labeled":[], # do not extract sublabels under these labels
        "new_labels":["title", "fragment"],
        "keep_attributes":["labelname"],
        "switch_type":True, # manual_label -> auto_label
        "use_simplified":True, # <auto_label labelname="title"> -> <title>
        }
        distribution=[0.4, 0.4]
        
    if out_version == "v1.2":
        sublabel_config = {
        "parent":["secondary sources"], # only extract sublabels under these parents
        "already_labeled":["title", "fragment"], # do not extract sublabels under these labels
        "new_labels":["source", "authors"],
        "keep_attributes":["labelname"], 
        "switch_type":True, # manual_label -> auto_label
        "use_simplified":True, # <auto_label labelname="title"> -> <title>
        }
        distribution=[0.4, 0.4]

    if out_version == "v1.3":
        sublabel_config = {
            "parent":["decision", "legislation"], # only extract sublabels under these parents
            "already_labeled":["title", "fragment", "source", "authors"], # do not extract sublabels under these labels
            "new_labels":["citation"],
            "keep_attributes":["labelname"], 
            "switch_type":True, # manual_label -> auto_label
            "use_simplified":True, # <auto_label labelname="title"> -> <title>
        }
        distribution=[0.8]

    return sublabel_config, distribution

def main():

    versions = ["v1.1", "v1.2", "v1.3"]
    min_tokens, fs_min_tokens, fs_mode, model_name, n_few_shot, prompt_version, prompt_path, sublabel_definitions, with_context = get_hyperparameters()

    # Modified output directory to distinguish context version
    base_dir = fr"{project_root}\data\Documents_Annotés\llm\TEST_PARACHUNKER_p{prompt_version}_c{min_tokens}_fs{fs_mode}-{n_few_shot}_m{model_name}"
    output_dir = base_dir

    # Mapping: version -> input_suffix (what to load)
    version_input_map = {
        "v1.1": "",        # Load original files (no suffix)
        "v1.2": "_v1.1",   # Load v1.1 files
        "v1.3": "_v1.2"    # Load v1.2 files
    }

    for version in versions:
        print(f"\n{'='*60}")
        print(f"Processing version: {version}")
        print(f"{'='*60}\n")

        input_suffix = version_input_map[version]
        
        # Load input files for this version
        input_files = {}
        
        for entry in os.listdir(base_dir):
            if not entry.lower().endswith(".html"):
                continue
            
            name, ext = os.path.splitext(entry)
            
            # Check if this file matches the input pattern for this version
            if input_suffix == "":
                # For v1.1, load files without version suffix (v1.0/original)
                # Exclude files with version suffixes
                if not any(name.lower().endswith(suffix) for suffix in ["_v1.1", "_v1.2", "_v1.3"]):
                    base_name = name
                    full_path = os.path.join(base_dir, entry)
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            input_files[base_name] = f.read()
                    except UnicodeDecodeError:
                        with open(full_path, "r", encoding="latin-1") as f:
                            input_files[base_name] = f.read()
            else:
                # For v1.2 and v1.3, load files with specific version suffix
                if name.lower().endswith(input_suffix.lower()):
                    base_name = name[:-len(input_suffix)]  # Remove version suffix
                    full_path = os.path.join(base_dir, entry)
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            input_files[base_name] = f.read()
                    except UnicodeDecodeError:
                        with open(full_path, "r", encoding="latin-1") as f:
                            input_files[base_name] = f.read()
        
        if not input_files:
            print(f"   ⚠ No input files found for {version}")
            print(f"      Expected files with suffix: '{input_suffix}' (or no suffix for v1.1)")
            print(f"      Skipping version {version}...")
            continue
        
        print(f"   ✓ Loaded {len(input_files)} input files for {version}")
        
        # Build set of already processed output files for this version
        existing_processed = set()
        for entry in os.listdir(output_dir):
            if not entry.lower().endswith(".html"):
                continue
            stem, _ = os.path.splitext(entry)
            if stem.lower().endswith(f"_{version}".lower()):
                original_name = stem[:-len(f"_{version}")]
                existing_processed.add(original_name.lower())
        
        print(f"   ✓ Found {len(existing_processed)} already processed files for {version}")

        # Process each file
        for filename, html_content in input_files.items():
            clean_filename = re.sub(r'_v\d+(\.\d+)*$', '', filename, flags=re.IGNORECASE)
            if clean_filename.lower() in existing_processed:
                print(f"   ✓ Skipping {filename} (output {version} already exists)")
                continue

            print(f"\n   → Processing: {filename}.html → {clean_filename}_{version}.html")
            #======================= MAIN STEPS ==========================#

            
            tokens = tokenize(html_content)

            sublabel_config, distribution = get_label_config(version)

            selected_few_shot_examples = main_few_shot_selection(filename=filename, sublabel_config=sublabel_config, distribution=distribution, n_few_shot=n_few_shot)

            
            model = GPTAssistant(model_name, temperature=1)



            processed_label = process_labels(
                model=model,
                tokens=tokens,
                sublabel_config=sublabel_config,
                few_shot_examples=selected_few_shot_examples,
                prompt_path=prompt_path,
                sublabel_definitions=sublabel_definitions,
                output_dir=output_dir,
                filename=filename,
                with_context=with_context,
            )


            processed_html_content = main_post_processing(processed_label=processed_label, html_content=html_content, original_tokens=tokens)
            # ---------- Save processed HTML to file ----------
            with open(fr"{output_dir}\{clean_filename}_{version}.html", 'w', encoding='utf-8') as f:
                f.write(processed_html_content)
            print(f"   ✓ Processed HTML saved to: {output_dir}")



if __name__ == "__main__":
    main()
