from utils import check_auto_label_consistency, apply_operations_safe
from utils import get_prompt_processing, get_prompt_fallback_consistency, get_prompt_fallback_hallucination, clean_token_manual_label
from utils import distance_lists_auto_label
from utils import tokenize, decode
from utils import clean_tokens

from tqdm import tqdm
import os
import json


def process_chunks(model, token_chunks, few_shot_examples=None, output_dir=None, filename=None):
    """
    Process a list of sentences using AI model to enhance them.
    
    Args:
        sentences: List of sentences to process
        few_shot_examples: List of tuples (input, expected_output) for few-shot learning
        model_name: Name of the AI model to use
    
    Returns:
        list: Enhanced sentences
    """
    system_prompt, user_prompt_template = get_prompt_processing(few_shot_examples)
    
    processed_chunks = []

    history = []
    
    print(f"   ✓ Processing {len(token_chunks)} texts with an LLM...")
    print(f"   ✓ Using {len(few_shot_examples) if few_shot_examples else 0} few-shot examples")
    
    for chunk in tqdm(token_chunks, desc="processing document..."):

       
        
        cleaned_chunk = []
        for token in chunk:

             # ------ 1. TOKENS -> CLEANED TOKENS ------
            cleaned_token = clean_token_manual_label(token, auto_transformation=False) #Keep manual_label tags
            cleaned_chunk.append(cleaned_token)
        
        # ------ 2. TOKENS -> TEXT ------
        text = decode(cleaned_chunk)


        # ------ 3. TEXT -> PROCESS TEXT ------
        user_prompt = user_prompt_template.format(text=text)
        processed_chunk = model.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        processed_chunk_tokens = tokenize(processed_chunk)
        
        # ----- 4. FALLBACK ERROR CORRECTION -----
        _, operations = distance_lists_auto_label(cleaned_chunk, processed_chunk_tokens) 

        processed_chunk_tokens_corrected = apply_operations_safe(processed_chunk_tokens, operations) # Get the original text while keeping the auto_label tags
        

        # ------ 5. HALLUCINATION VERIFICATION ------
        hal_test = clean_tokens(cleaned_chunk, normalize=True, keep_manual_label=True) == clean_tokens(processed_chunk_tokens_corrected, normalize=True, keep_manual_label=True)


        if not hal_test:
            print("Hallucination test failed")
             # ------ 5. FALLBACK ------
            fall_back_system_promt, fall_back_user_prompt_template = get_prompt_fallback_hallucination()
            user_prompt = fall_back_user_prompt_template.format(text_pair=f"Original Text :\n\n<<<ORIGINAL_TEXT>>>{text}<<<END_ORIGINAL_TEXT>>>\n\nAnnotated Text with errors:\n\n<<<ANNOTATED_TEXT>>>{processed_chunk}<<<END_ANNOTATED_TEXT>>>\n\n")
            processed_chunk = model.generate(
                system_prompt=fall_back_system_promt,
                user_prompt=user_prompt
            )
            processed_chunk_tokens = tokenize(processed_chunk)

            # ----- 4. FALLBACK ERROR CORRECTION -----
            _, operations = distance_lists_auto_label(cleaned_chunk, processed_chunk_tokens) 

            processed_chunk_tokens_corrected = apply_operations_safe(processed_chunk_tokens, operations) # Get the original text while keeping the manual_label tags

            # ------ 5. HALLUCINATION VERIFICATION ------
            hal_test = clean_tokens(cleaned_chunk, normalize=True, keep_manual_label=True) == clean_tokens(processed_chunk_tokens_corrected, normalize=True, keep_manual_label=True)

            if not hal_test:
                history.append(("Double Hallucination Fail", processed_chunk))
                print("Hallucination test failed again")
                processed_chunks.append(chunk)
                continue
            


        # ------ 6. CONSISTENCY CHECK -----
        const_check = check_auto_label_consistency(processed_chunk_tokens_corrected)

        if not const_check:
            print("Consistency check failed")
            # ------ 5. FALLBACK ------
            fall_back_system_promt, fall_back_user_prompt_template = get_prompt_fallback_consistency()
            user_prompt = fall_back_user_prompt_template.format(text_pair=f"Original Text :\n\n<<<ORIGINAL_TEXT>>>{text}<<<END_ORIGINAL_TEXT>>>\n\nAnnotated Text with errors:\n\n<<<ANNOTATED_TEXT>>>{processed_chunk}<<<END_ANNOTATED_TEXT>>>\n\n")
            processed_chunk = model.generate(
                system_prompt=fall_back_system_promt,
                user_prompt=user_prompt
            )
            processed_chunk_tokens = tokenize(processed_chunk)

            # ----- 4. FALLBACK ERROR CORRECTION -----
            _, operations = distance_lists_auto_label(cleaned_chunk, processed_chunk_tokens) 

            processed_chunk_tokens_corrected = apply_operations_safe(processed_chunk_tokens, operations) # Get the original text while keeping the manual_label tags

            # ------ 5. HALLUCINATION VERIFICATION ------
            hal_test = clean_tokens(cleaned_chunk, normalize=True, keep_manual_label=True) == clean_tokens(processed_chunk_tokens_corrected, normalize=True, keep_manual_label=True)

            if not hal_test:
                history.append(("Hallucination Fail after Consistency Fallback", processed_chunk))
                print("Hallucination test failed after consistency fallback")
                processed_chunks.append(chunk)
                continue
            
            # ------ 6. CONSISTENCY CHECK -----
            const_check = check_auto_label_consistency(processed_chunk_tokens_corrected)

            if not const_check:
                history.append(("Double Consistency Fail", processed_chunk))
                print("Consistency check failed again")
                processed_chunks.append(chunk)
                continue


        if hal_test and const_check:
            
            processed_chunks.append(processed_chunk_tokens_corrected)
            history.append(("Success", processed_chunk))

                
            

    
    print(f"   ✓ AI processing completed")

    # Save it in a json file for reference
    if output_dir and filename:
        json_path = os.path.join(output_dir, f"history_{filename}.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(history, json_file, ensure_ascii=False, indent=4)
            print(f"   ✓ Enhanced texts saved to: {json_path}")
        except Exception as e:
            print(f"   ✗ Error saving enhanced texts: {e}")

    
    # Save results
    with open(f"{output_dir}/processed_chunks_{filename}.json", "w", encoding="utf-8") as f:
        json.dump(processed_chunks, f, indent=4, ensure_ascii=False)

    return processed_chunks