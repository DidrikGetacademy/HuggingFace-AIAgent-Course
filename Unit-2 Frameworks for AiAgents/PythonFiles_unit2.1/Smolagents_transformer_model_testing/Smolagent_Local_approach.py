#1. INSTALL pip install smolagents transformers torch
from smolagents import TransformersModel

Model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

model = TransformersModel(model_id=Model_id, device="cuda")

response = model(
    [{"role": "user", "content": [{"type": "text", "text": "how can i learn python"}]}]
)


if __name__ == "__main__":
    print(response)

