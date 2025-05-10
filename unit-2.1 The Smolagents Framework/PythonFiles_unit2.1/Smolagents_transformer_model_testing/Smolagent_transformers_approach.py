from smolagents import TransformersModel

model = TransformersModel(model_id="HuggingFaceTB/SmolLM-135M-Instruct")

print(model([{"role": "user", "content": [{"type": "text", "text": "Ok!"}]}], stop_sequences=["great"]))





#class smolagents.TransformersModel
#Parameters
#model_id (str) — The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub. For example, "Qwen/Qwen2.5-Coder-32B-Instruct".
#device_map (str, optional) — The device_map to initialize your model with.
#torch_dtype (str, optional) — The torch_dtype to initialize your model with.
#trust_remote_code (bool, default False) — Some models on the Hub require running remote code: for this model, you would have to set this flag to True.
#kwargs (dict, optional) — Any additional keyword arguments that you want to use in model.generate(), for instance max_new_tokens or device.
#**kwargs — Additional keyword arguments to pass to model.generate(), for instance max_new_tokens or device.