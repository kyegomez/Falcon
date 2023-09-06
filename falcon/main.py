import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


class Falcon:
    def __init__(
        self,
        *,
        model_id: str = None,
        temperature: float = None,
        top_p: float = None,
        max_new_tokens: int = None,
        quantized: bool = False,
        system_prompt: str = None
    ):
        super().__init__()

        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.quantized = quantized
        self.system_prompt = system_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        if self.quantized:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
    
    def run(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        output = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens
        )

        output = output[0].to("cuda")
        print(self.tokenizer.decode(output))

    def chat(self, message, history, system_prompt):
        prompt = ""
        system_prompt = system_prompt or self.system_prompt
        if system_prompt:
            prompt += f"System: {system_prompt}\n"
        for user_prompt, bot_response in history:
            prompt += f"User: {user_prompt}\n"
            prompt += f"Falcon: {bot_response}\n"
            prompt += f"User: {message}\nFalcon:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to("cuda")

        output = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens
        )
        output = output[0].to("cuda")
        print(self.tokenizer.decode(output))

    
        
    

