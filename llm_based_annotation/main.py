import os
from utils_extraction import extract_body, tokenize, clean_tokens, decode
from utils_extraction import chunk_tokens, flatten_token_chunks
from utils_extraction import extract_few_shot_examples
from utils_extraction import select_few_shot 
from utils_extraction import merge_tokens_with_auto_labels, add_attributes_to_auto_labels, compare_html_allow_auto_labels, correct_tokens_brackets, check_tokens_brackets
from models import GPTAssistant
from utils_extraction import process_chunks
from utils_extraction.html_utils import clean_html_formatting
import json

project_root = r"C:\Users\zakga\OneDrive\Documents\code\LeREaD_annotation_process"



def main():

    # ---------- Define Hyperparameters ----------
    min_tokens = 500
    fs_min_tokens = 100
    fs_mode = "selected"  # "random" or "selected"
    model_name = "gpt-5.2"

    n_few_shot = 30  # Number of few-shot examples to use

    prompt_version = "2"
    cot = False
    if cot :
        prompt_path = fr"{project_root}\llm_based_annotation\utils_extraction\prompts\simplified_parent_extraction_cot v{prompt_version}.txt"
    else :
        prompt_path = fr"{project_root}\llm_based_annotation\utils_extraction\prompts\simplified_parent_extraction v{prompt_version}.txt"



    # File paths
    filename = "1997CanLII16226_ONCA" #"1989CanLII1415ONCA" #"2021QCCA1675" #"1997CanLII16226_ONCA"
    round = "ronde_2"
    anno = "llm"
    version = "v1.0"

    try:
        ext = "html"
        html_path = fr"{project_root}\data\Document_Échantillon_Initial\{round}\plain_html_arbre_balise\{filename}.{ext}"
        with open(html_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        

    except Exception as e:
        ext = "htm"
        html_path = fr"{project_root}\data\Document_Échantillon_Initial\{round}\plain_html_arbre_balise\{filename}.{ext}"
        # Read HTML file
        with open(html_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

    print(f"   ✓ HTML file loaded: {html_path}")
    if fs_mode == "selected" :
        output_dir = fr"{project_root}\data\Documents_Annotés\{anno}\{filename}\v_prompt_{prompt_version}_{min_tokens}_{fs_mode}_{n_few_shot}_gpt5.2_chunk2_subdef2"
    else :
        output_dir = fr"{project_root}\data\Documents_Annotés\{anno}\{filename}\v_prompt_{prompt_version}_{min_tokens}_{fs_min_tokens}_{n_few_shot}_gpt5.2_chunk2_subdef2"
    os.makedirs(output_dir, exist_ok=True)




if __name__ == "__main__":
    main()