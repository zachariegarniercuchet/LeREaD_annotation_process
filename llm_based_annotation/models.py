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
    

def check_accessibility():
    client = OpenAI()

    models = client.models.list()
    for m in models.data:
        print(m.id)

if __name__ == "__main__":

    """
    assistant = GPTAssistant(model_name="gpt-4.1", temperature=0.5)
    system_prompt = "You are a helpful assistant."
    user_prompt = "Explain the theory of relativity in simple terms."
    response = assistant.generate(system_prompt, user_prompt)
    print(response)
    """

    check_accessibility()