from openai import OpenAI
import os

class GPTAssistant:
    def __init__(self, model_name : str, temperature : float = 0.3):
        self.model = model_name
        self.temperature = temperature

    def generate(self, system_prompt, user_prompt):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model= self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature
            )

        return response.choices[0].message.content

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

MODELS_DIR = "/home/zagar/projects/def-azouaq/share/models"

class QwenAssistant:
    def __init__(self, model_name: str = "Qwen2.5-7B-Instruct", temperature: float = 0.3):
        self.temperature = temperature
        model_path = os.path.join(MODELS_DIR, model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        # Decode only the newly generated tokens (strip the prompt)
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)




    

def check_accessibility():
    client = OpenAI()

    models = client.models.list()
    for m in models.data:
        print(m.id)

if __name__ == "__main__":

    
    assistant = GPTAssistant(model_name="gpt-5.2", temperature=1)
    system_prompt = "You are a helpful assistant."
    user_prompt = "Explain the theory of relativity in simple terms."
    response = assistant.generate(system_prompt, user_prompt)
    print(response)
    

    check_accessibility()