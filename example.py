from falcon.main import Falcon


falcon = Falcon(
    temperature=0.5, 
    top_p=0.9, 
    max_new_tokens=500,
    quantized=True,
    system_prompt=""
)

prompt = "What is the meaning of the collapse of the wave function?"

result = falcon.run(prompt=prompt)
print(result)