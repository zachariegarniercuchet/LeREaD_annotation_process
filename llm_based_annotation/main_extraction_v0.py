import os
from utils_extraction import extract_body, tokenize, clean_tokens, decode, is_auto_label_tag, is_tag_token
from utils_extraction import flatten_token_chunks, merge_sentences_with_heuristics_tokens
from utils_extraction import prepare_label_tokens
from utils_extraction import merge_tokens_general, add_attributes_to_auto_labels, compare_html_allow_auto_labels, correct_tokens_brackets, check_tokens_brackets
from models import GPTAssistant
from utils_extraction import process_chunks
from utils_extraction import clean_html_formatting
from utils_extraction import select_few_shot_for_all_chunks


import json
import spacy
import re

project_root = r"C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process"


def main_chunk_html(html_content, min_tokens=500, method="sentence"):

    if method == "sentence":
        return chunk_html_by_sentence(html_content, min_tokens=min_tokens)
    elif method == "paragraph":
        return chunk_html_by_paragraph(html_content, min_tokens=min_tokens)
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'sentence' or 'paragraph'.")
    


def chunk_html_by_sentence(html_content, min_tokens=500):
    body_content = extract_body(html_content)


    # ---------- Tokenize body content ----------
    tokens = tokenize(body_content)

    # ---------- Clean tokens ----------
    normalized_cleaned_tokens = clean_tokens(html_tokens=tokens, normalize=True, keep_manual_label=True, keep_bookmarks=True)

    

    nlp = spacy.load("en_core_web_trf")

    doc = nlp(decode(normalized_cleaned_tokens))
    initial_sentences = [sent.text for sent in doc.sents]

    initial_sentences_token = [tokenize(sent) for sent in initial_sentences]
    flat_initial_sentences = flatten_token_chunks(initial_sentences_token, separator="<sep>")


    is_sep_tag = lambda token: token == '<sep>'

    # Example: merge normalized_cleaned_tokens (original, no <sep>) 
    # with flat_token_sentence_chunks (derived, with <sep>)

    print(f"Original tokens: {len(normalized_cleaned_tokens)} (no <sep>)")
    print(f"Derived tokens: {len(flat_initial_sentences)} (with <sep>)")

    # This assumes flat_token_sentence_chunks is normalized_cleaned_tokens + <sep> insertions
    corrected_initial_sentences = merge_tokens_general(
        original_tokens=normalized_cleaned_tokens,
        derived_tokens=flat_initial_sentences,
        is_protected_func=is_sep_tag,
        log=False
    )

    print(f"\nResult: {len(corrected_initial_sentences)} tokens")
    print(f"Number of <sep> tags: {corrected_initial_sentences.count('<sep>')}")

    # Apply heuristic-based merging with sequential citation detection
    CITATION_THRESHOLD = 25  # Combined density threshold (%) - adjust based on the graphs above

    flat_token_sentence_chunks = merge_sentences_with_heuristics_tokens(corrected_initial_sentences, citation_threshold=CITATION_THRESHOLD, min_tokens=min_tokens)
    print(f"Initial sentences: {len(corrected_initial_sentences)}, After merging: {len(flat_token_sentence_chunks)}")
    print(f"Using citation threshold: {CITATION_THRESHOLD}% (combined period + number density)")
    print(f"Gap tolerance: 3 consecutive sentences below threshold to end citation section")
    print(f"Note: Only the FIRST citation section is detected; all subsequent sentences are not citations")


    token_chunks =  []
    current_chunk = []

    for token in flat_token_sentence_chunks:

        if token != "<sep>":
            current_chunk.append(token)
        else:
            token_chunks.append(current_chunk)
            current_chunk =  []
    token_chunks.append(current_chunk)

    # Verify: merged should equal normalized_cleaned_tokens if ignoring <sep> tag

    if flatten_token_chunks(token_chunks) == normalized_cleaned_tokens:
        print("✓ Perfect match! Derived was indeed original + <sep> insertions")
    else:
        print("⚠ Some differences exist beyond <sep> insertions")
        # Show first difference
        for i, (m, d) in enumerate(zip(flat_token_sentence_chunks, flat_token_sentence_chunks)):
            if m != d:
                print(f"  First diff at index {i}: merged='{m}' vs derived='{d}'")
                break
        assert "Difference detected"

    return token_chunks

def chunk_html_by_paragraph(html_content, min_tokens=500):
    """
    Paragraph-based chunker. Uses BeautifulSoup to extract leaf block elements,
    preserving their HTML, then tokenizes + cleans each paragraph, and merges
    consecutive paragraphs until min_tokens is reached.

    Returns: List[List[token]] — same format as the sentence method.
    """
    from bs4 import BeautifulSoup

    body_content = extract_body(html_content)

    # ------------------------------------------------------------------ #
    # 1. Extract leaf block elements (same logic as batch_paragraphs)     #
    # ------------------------------------------------------------------ #
    soup = BeautifulSoup(body_content, "html.parser")

    def get_leaf_blocks(tag):
        leaf_blocks = []
        for child in tag.find_all(
            ["p", "li", "blockquote", "pre", "h1", "h2", "h3", "h4", "h5", "h6"]
        ):
            # Only keep blocks that don't contain other block-level elements
            if not child.find(["p", "li", "blockquote", "pre"]):
                leaf_blocks.append(child)
        return leaf_blocks

    leaf_blocks = get_leaf_blocks(soup)

    # ------------------------------------------------------------------ #
    # 2. Tokenize + clean each paragraph, keeping original HTML           #
    # ------------------------------------------------------------------ #
    paragraph_token_lists = []  # List[List[token]]

    for block in leaf_blocks:
        # str(block) preserves the full HTML of the element (tags included)
        block_html = str(block)

        raw_tokens = tokenize(block_html)
        cleaned_tokens = clean_tokens(
            html_tokens=raw_tokens,
            normalize=True,
            keep_manual_label=True,
            keep_bookmarks=True,
        )

        if cleaned_tokens:
            paragraph_token_lists.append(cleaned_tokens)

    # ------------------------------------------------------------------ #
    # 3. Merge consecutive paragraphs until min_tokens is reached         #
    # ------------------------------------------------------------------ #
    token_chunks = []
    current_chunk = []

    for para_tokens in paragraph_token_lists:
        if not current_chunk:
            # Always start a new chunk with the current paragraph
            current_chunk = list(para_tokens)
        elif len(current_chunk) >= min_tokens:
            # Current chunk is already big enough — flush and start fresh
            token_chunks.append(current_chunk)
            current_chunk = list(para_tokens)
        else:
            # Current chunk is still too small — keep accumulating
            current_chunk.extend(para_tokens)

    # Flush the last chunk
    if current_chunk:
        token_chunks.append(current_chunk)

    print(f"Paragraph chunks: {len(token_chunks)}")
    print(f"Chunk sizes (tokens): {[len(c) for c in token_chunks]}")

    return token_chunks



def main_few_shot_selection(filename, label_config, n_few_shot=30, fs_json_path=None, random_seed=None):
    
    import random
    use_random_mode = random_seed is not None
    
    if use_random_mode:
        random.seed(random_seed)
        if fs_json_path is None:
            fs_json_path = fr"{project_root}\few_shot_selection_tool\second_selected\examples_selected_45_with_sources_fixed_spacing_manual_label.json"
    else:
        if fs_json_path is None:
            fs_json_path = fr"{project_root}\few_shot_selection_tool\second_selected\combined_v3_with_sources_manual_label.json"
    
    # Load the few-shot examples JSON
    with open(fs_json_path, 'r', encoding='utf-8') as file:
        fs_data = json.load(file)
    print(f"   ✓ Loaded {len(fs_data)} examples from: {fs_json_path}")
    print(f"   ✓ Selection mode: {'random' if use_random_mode else 'manual'}")
    
    # Get all unique source files and print them
    source_files = sorted(list(set([item.get('source_file', 'unknown') for item in fs_data])))
    print(f"\n   Source files in few-shot collection:")
    for sf in source_files:
        count = sum(1 for item in fs_data if item.get('source_file') == sf)
        print(f"      - {sf} ({count} examples)")
    
    # Apply filters:
    # 1. Mask filter: Exclude examples from the same document being annotated (avoid data leakage)
    # 2. Selection filter: Either random selection or manual "selected" flag
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

        # Filter 2: Apply selection filter based on mode
        if use_random_mode:
            # In random selection mode, randomly select from the whole pool (after masking)
            # Keep item if random number is <= probability (n_few_shot / len(fs_data))
            if random.random() > (n_few_shot / len(fs_data)):
                excluded_not_selected += 1
                continue
        else:
            # In manual selection mode, only keep examples where "selected" == true
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
    
    #print(f"\n   ✓ Filtering results:")
    #print(f"      - Excluded (same document): {excluded_same_doc}")
    #print(f"      - Excluded (not selected): {excluded_not_selected}")
    #print(f"      - Retained: {len(filtered_examples)}")
    
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
    
    
    
    #print(f"\n   ✓ Simplifying outputs to parent-level extraction...")
    simplified_examples = []
    for ex in selected_examples_dicts:
        simplified_output = decode(prepare_label_tokens(tokenize(ex['output']), label_config=label_config))
        simplified_examples.append((ex['input'], simplified_output))
    
    # Convert to list of tuples (input, output)
    selected_few_shot_examples = simplified_examples
    
    #print(f"   ✓ Selected {len(selected_few_shot_examples)} few-shot examples for processing.")
    
    return selected_few_shot_examples


def main_post_processing(processed_chunks, html_content):
    # Processed_chunks is a list of lists of tokens, we need to flatten it to get a single list of tokens for the whole document
    processed_tokens_flat = flatten_token_chunks(processed_chunks)


    # Read in parallel the original tokens and the processed tokens. Always prefer the original tokens, but if there is an auto_label token in the processed tokens, 
    # we want to keep it and merge it with the original tokens. 
    # This way we can keep the original formatting and structure of the document while adding the auto_labels extracted by the model.
    original_tokens = tokenize(html_content)
    processed_html_content_tokens = merge_tokens_general(
        original_tokens=original_tokens,
        derived_tokens=processed_tokens_flat,
        is_protected_func=lambda tok: is_auto_label_tag(tok) != 0,
        is_opening_protected_func=lambda tok: is_auto_label_tag(tok) == 1,
        is_tag_token_func=lambda tok: is_tag_token(tok),
        log=False
        )

    # check of the final processed_html_content with the original HTML, ignoring the auto_label tags which are not present in the original HTML but only in the processed one.
    comparison_result = compare_html_allow_auto_labels(decode(processed_html_content_tokens), html_content)
    assert comparison_result, "The processed HTML content does not match the original HTML content when ignoring auto_label tags. Please check the merging and post-processing steps for errors."


    
    # This merging process can sometimes create some formatting issues with brackets, we need to correct them to get a valid HTML structure.
    processed_html_content_tokens_corrected = correct_tokens_brackets(processed_html_content_tokens)
    assert check_tokens_brackets(processed_html_content_tokens_corrected), "The brackets in the merged tokens are not balanced. Please check the merging and bracket correction steps for errors."


    # The correction of the brackets can sometimes create some redoundant or useless formatting  with the HTML, we need to clean it to compare it with the original.
    processed_html = decode(processed_html_content_tokens_corrected)
    processed_html_cleaned = clean_html_formatting(processed_html)
    print(f"\nMerged HTML length: {len(processed_html_cleaned)}")

    
    # This step is just to ensure a good visualisation of HTMLLabelizer and to add the necessary attribute to stay consistent with the label scheme
    processed_html_content = add_attributes_to_auto_labels(processed_html_cleaned)

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
    fs_mode = "pattern"  # "random" or "selected" or "pattern"
    model_name = "gpt-5.2"


    #if not pattern selected, please set n_dynamic = 0. If pattern selected, we will have n_general fixed examples + n_dynamic chunk-based examples.
    n_general = 5
    n_dynamic = 25

    n_few_shot = (n_general, n_dynamic) 
    
    all_in_one_go = True  # Whether to run the entire extraction process in one go 

    fall_back = True  # Whether to implement a fall-back mechanism in case of verification failure (e.g., hallucination, consistency, label scheme errors)

    prompt_version = "2"
    cot = False
    if cot :
        prompt_path = fr"{project_root}\llm_based_annotation\utils_extraction\prompts\simplified_parent_extraction_cot v{prompt_version}.txt"
    else :
        prompt_path = fr"{project_root}\llm_based_annotation\utils_extraction\prompts\simplified_parent_extraction v{prompt_version}.txt"
    
    if all_in_one_go:
        prompt_path = fr"{project_root}\llm_based_annotation\utils_extraction\prompts\allinone_v{prompt_version}.txt"

    chunking_method = "paragraph"  # "sentence" or "paragraph"


    if not all_in_one_go:
        # Define label_config 
        label_config = {
            "keep_attributes": ["labelname"],  # extraction only, no disambiguation
            "switch_type": True,  # manual_label -> auto_label
            "use_simplified": True,  # <auto_label labelname="title"> -> <title>
            "keep_labels": ["decision", "legislation", "secondary sources"]
        }
    else:
        label_config = {
            "keep_attributes": ["labelname"],  # extraction only, no disambiguation
            "switch_type": True,  # manual_label -> auto_label
            "use_simplified": True,  # <auto_label labelname="title"> -> <title>
            "keep_labels": ["decision", "legislation", "secondary sources", "citation", "source", "authors", "title", "fragment"]  # only keep these 4 parent-level labels, remove all sublabels
        }

    return min_tokens, fs_min_tokens, fs_mode, model_name, n_few_shot, prompt_version, prompt_path, cot, label_config, chunking_method, fall_back

def main():

    min_tokens, _, fs_mode, model_name, n_few_shot, prompt_version, prompt_path, cot, label_config, chunking_method, fall_back = get_hyperparameters()

    if n_few_shot[1] > 0:
            dynamic_few_shot = True
    else:
        dynamic_few_shot = False


    source_dir = fr"{project_root}\data\final\Original"
    #source_dir = fr"{project_root}\data\Document_Échantillon_Initial\ronde_3\plain_html_arbre_balise"
    files = get_all_html_files_in(source_dir)
    print(f"✓ Loaded {len(files)} files from: {source_dir}")

    if n_few_shot[1] > 0:
        output_dir = fr"{project_root}\data\Documents_Annotés\llm\F+_PARACHUNKER_ALLINONE_p{prompt_version}_c{min_tokens}_fs{fs_mode}-{n_few_shot[0]}-{n_few_shot[1]}_m{model_name}"
    else:
        output_dir = fr"{project_root}\data\Documents_Annotés\llm\F+_PARACHUNKER_ALLINONE_p{prompt_version}_c{min_tokens}_fs{fs_mode}-{n_few_shot[0]}_m{model_name}"

    os.makedirs(output_dir, exist_ok=True)

    # Build a set of already processed filenames (case-insensitive), based on *_v1.0.html
    existing_processed = set()
    for entry in os.listdir(output_dir):
        if not entry.lower().endswith(".html"):
            continue
        stem, _ = os.path.splitext(entry)
        if stem.lower().endswith("_v1.0"):
            original_name = stem[:-5]  # remove "_v1.0"
            existing_processed.add(original_name.lower())

    for filename, html_content in files.items() :
        if filename.lower() in existing_processed:
            print(f"   ✓ Skipping {filename}.html (already processed)")
            continue

        
        token_chunks = main_chunk_html(html_content, min_tokens=min_tokens, method=chunking_method)

        if fs_mode == "random":
            selected_few_shot_examples = main_few_shot_selection(filename=filename, n_few_shot=sum(n_few_shot), 
                                                                 #fs_json_path=fr"{project_root}\few_shot_selection_tool\second_selected\examples_selected_45_clean_with_sources_fixed_spacing_manual_label.json",
                                                                 random_seed=42,
                                                                 label_config=label_config)  # Set a random seed for reproducibility if using random selection mode 
        elif fs_mode == "selected":
            selected_few_shot_examples = main_few_shot_selection(filename=filename, n_few_shot=sum(n_few_shot), 
                                                                 fs_json_path=fr"{project_root}\few_shot_selection_tool\second_selected\examples_selected_45_clean_with_sources_fixed_spacing_manual_label.json",
                                                                 random_seed=None,
                                                                 label_config=label_config)  # Set to None for manual selection mode
        elif fs_mode == "pattern":
            
            if dynamic_few_shot:
                selected_few_shot_examples = select_few_shot_for_all_chunks(
                    token_chunks_par = token_chunks,
                    filename = filename,
                    label_config = label_config,
                    n_general = n_few_shot[0],
                    n_dynamic = n_few_shot[1],
                    general_json_path = fr"{project_root}\few_shot_selection_tool\greedy_set_coverage_rejected_corrected.json",
                    dynamic_json_path = fr"{project_root}\few_shot_selection_tool\second_selected\examples_selected_45_with_sources_fixed_spacing_manual_label_with_patterns2.json",
                    random_seed = None,
                ) # multiple examples (input, output) by chunk. selected_few_shot_examples[0][0][0] is the input of the first example of the first chunk.

                with open(f"{output_dir}\\selected_few_shot_examples_{filename}.json", "w", encoding="utf-8") as f:
                    json.dump(selected_few_shot_examples, f)

            else:
                selected_few_shot_examples = main_few_shot_selection(filename=filename, n_few_shot=sum(n_few_shot), 
                                                                 fs_json_path=fr"{project_root}\few_shot_selection_tool\greedy_set_coverage_rejected_corrected.json",
                                                                 random_seed=None,
                                                                 label_config=label_config)  # Set to None for manual selection mode

        model = GPTAssistant(model_name, temperature=1)

        print("------------------ PROMPT FROM FILE ------------------")
        print(prompt_path)
        print("------------------------------------------------------")

        

        processed_chunks = process_chunks(
            model=model,
            token_chunks=token_chunks,
            process_prompt_path=prompt_path,
            label_config=label_config,
            few_shot_examples=selected_few_shot_examples,
            output_dir=output_dir,
            filename=filename,
            cot = cot,
            dynamic_few_shot_selection = dynamic_few_shot,
            fall_back=fall_back
            )
        
        with open(f"{output_dir}\\processed_chunks.json", "w") as f:
            json.dump(processed_chunks, f)


        processed_html_content = main_post_processing(processed_chunks, html_content)
        # ---------- Save processed HTML to file ----------
        with open(fr"{output_dir}\{filename}_v1.0.html", 'w', encoding='utf-8') as f:
            f.write(processed_html_content)
        print(f"   ✓ Processed HTML saved to: {output_dir}")



if __name__ == "__main__":
    main()