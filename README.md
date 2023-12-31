[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Simple Falcon
A simple package for leveraging Falcon 180B and the HF ecosystem's tools, including training/inference scripts, safetensors, integrations with bitsandbytes, PEFT, GPTQ, assisted generation, RoPE scaling support, and rich generation parameters.


## Installation

You can install the package using pip

```bash
pip3 install simple-falcon
```
---

# Usage

```python
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
```

# Documentation

The Falcon class provides a convenient interface for conversational agents based on the transformers architecture. It facilitates both single-turn and multi-turn conversations with pre-trained models and allows users to customize certain inference settings such as `temperature`, `top_p`, and token generation limits. Furthermore, it can leverage quantized models for faster performance.

### Purpose

The main purpose of the Falcon class is to:
- Make it easy to initiate and run generative language models.
- Provide efficient conversation interfaces with customization.
- Support both regular and quantized models for better performance.
- Manage conversational history in multi-turn scenarios.

### Class Definition:

```python
class Falcon:
    def __init__(
        self,
        *,
        model_id: str = "tiiuae/falcon-180B",
        temperature: float = None,
        top_p: float = None,
        max_new_tokens: int = None,
        quantized: bool = False,
        system_prompt: str = None
    ):
```

#### Parameters:

- **model_id (str)**: Model identifier from the HuggingFace Model Hub. Default is "tiiuae/falcon-180B".
  
- **temperature (float, optional)**: Controls randomness in the Boltzmann distribution of model predictions. Higher values result in more randomness.
  
- **top_p (float, optional)**: Nucleus sampling: Restricts sampling to the top tokens summing up to this cumulative probability.
  
- **max_new_tokens (int, optional)**: Maximum number of tokens that can be generated in a single inference call.
  
- **quantized (bool)**: If set to `True`, the model loads in 8-bit quantized mode. Default is `False`.
  
- **system_prompt (str, optional)**: Initial system prompt to set the context for the conversation.

### Method Descriptions:

#### 1. run:

```python
def run(self, prompt: str) -> None:
```

Generates a response based on the provided prompt.

**Parameters**:
- **prompt (str)**: Input string to which the model responds.

**Returns**: None. The response is printed to the console.

#### 2. chat:

```python
def chat(self, message: str, history: list[tuple[str, str]], system_prompt: str = None) -> None:
```

Generates a response considering the conversation history.

**Parameters**:
- **message (str)**: User's current message to which the model will respond.
  
- **history (list[tuple[str, str]])**: Conversation history as a list of tuples. Each tuple consists of the user's prompt and the Falcon's response.
  
- **system_prompt (str, optional)**: Initial system prompt to set the context for the conversation.

**Returns**: None. The response is printed to the console.

### Usage Examples:

#### 1. Single-turn conversation:

```python
from falcon import Falcon
import torch

model = Falcon(temperature=0.8)
model.run("What is the capital of France?")
```

#### 2. Multi-turn conversation with history:

```python
from falcon import Falcon
import torch

model = Falcon(system_prompt="Conversational Assistant")
history = [
    ("Hi there!", "Hello! How can I assist you?"),
    ("What's the weather like?", "Sorry, I can't fetch real-time data, but I can provide general info.")
]
model.chat("Tell me a joke.", history)
```

#### 3. Using quantized models:

```python
from falcon import Falcon
import torch

model = Falcon(quantized=True)
model.run("Tell me about quantum computing.")
```

### Mathematical Representation:

The Falcon class essentially leverages the transformer-based generative language model for text generation. The mathematical process can be generalized as:

Given an input sequence \( x = [x_1, x_2, ... , x_n] \), the model predicts the next token \( x_{n+1} \) by:

\[ x_{n+1} = \arg \max P(x_i | x_1, x_2, ... , x_n) \]

Where:
- \( P \) is the probability distribution over the vocabulary generated by the model.
- The argmax operation selects the token with the highest probability.

### Additional Information:

- For best performance, it's recommended to use the Falcon class with CUDA-enabled devices. Ensure that your PyTorch setup supports CUDA.
  
- The Falcon class uses models from the HuggingFace model hub. Ensure you have an active internet connection during the first run as models will be downloaded.
  
- If memory issues arise, consider reducing the `max_new_tokens` parameter or using quantized models.

---

# License
MIT



