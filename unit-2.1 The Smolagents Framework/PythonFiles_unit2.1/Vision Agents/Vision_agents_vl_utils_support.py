from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer,AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

#Default: load the model on available devices
model = Qwen2VLForConditionalGeneration.from_pretrained(
    r"C:\Users\didri\Desktop\LLM-models\QWEN-VL-6B-INSTRUCT",
    torch_dtype="auto",
    device_map="auto"
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    r"C:\Users\didri\Desktop\LLM-models\QWEN-VL-6B-INSTRUCT",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)


#Default processer

#The default range for the number of visual tokens per image in the model is 4 -
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
            "type": "image",
            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

#Preperation for inference

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)
inputs= processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)

inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids,out_ids in zip(input.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True,clean_up_tokenization_spaces=False
)
print(output_text)
