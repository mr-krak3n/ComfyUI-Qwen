from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
import torch
import comfy.model_management as mm
import folder_paths
import os

model_directory = os.path.join(folder_paths.models_dir, "LLM")

def load_model(model_name, device, dtype, attention, load_in_4bit):
    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attention,
        **({ "quantization_config": quantization_config } if load_in_4bit else {}),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    return generator


def send_message(generator, history, max_new_tokens=512, seed=0, temperature=0.7, top_k=50, top_p=0.95, use_cache=False):
    set_seed(seed)
    kwargs = {
        'max_new_tokens' : max_new_tokens,
        'do_sample' : True,
        'temperature': temperature,
        'top_k' : top_k,
        'top_p' : top_p,
        'use_cache' : use_cache,
    }
    response = generator(history, **kwargs)
    return response[0]['generated_text'][-1]['content']


class QwenModel:
    def __init__(self, generator_pipeline):
        self.generator_pipeline = generator_pipeline


class QwenLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ([d for d in os.listdir(model_directory) if os.path.isdir(os.path.join(model_directory, d))],),
                "precision": (['fp16','bf16','fp32'],),
                "attention": (['flash_attention_2', 'sdpa', 'eager'], {"default": 'sdpa'}),
                "load_in_4bit": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("QWEN_MODEL",)
    FUNCTION = "load"
    CATEGORY = "Krak3n/qwen"

    def load(self, model_name, precision, attention, load_in_4bit):
        device = mm.get_torch_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        model_path = os.path.join(model_directory, model_name)

        generator = load_model(model_path, device, dtype, attention, load_in_4bit)

        return (QwenModel(generator_pipeline=generator),)


class DeepSeekResponseParser:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "response": ("STRING", {"forceInput": True}),
            }
        }
    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("think", "conclusion")
    FUNCTION = "parse"
    CATEGORY = "Krak3n/qwen"

    def parse(self, response):
        parts = response.split("</think>")
        conclusion = ""
        think = response
        if parts:
            if len(parts) > 0:
                conclusion = parts[1].strip()
            think = parts[0].replace("<think>", "").strip()
        return (think.strip(), conclusion.strip())


class QwenSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_model": ("QWEN_MODEL", {}),
                "system": ("STRING", {"multiline": True, "default": "Act like a prompt engineer for Stable Diffusion. You need to give me the most accturate prompt for my input. Don't introduce your message, give ONLY the prompt. Prompt must be around 150 words."}),
                "prompt": ("STRING", {"multiline": True, "default": "Give me a prompt for Stable Diffusion"}),
                "seed": ("INT", {"default": 0}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_cache": ("BOOLEAN", {"default": True}),
                "keep_model_loaded": ("BOOLEAN", {"default": False})
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "Krak3n/qwen"

    def generate(self, qwen_model, system, prompt, seed, max_new_tokens, temperature, top_k, top_p, use_cache, keep_model_loaded):
        model = qwen_model.generator_pipeline.model
        model.to(mm.get_torch_device())
        history = [
            {
                "role": "system", 
                "content": system
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        message = send_message(qwen_model.generator_pipeline, history, max_new_tokens, seed, temperature, top_k, top_p, use_cache)
        if not keep_model_loaded:
            model.to(mm.unet_offload_device())
            mm.soft_empty_cache()

        return (message, )

NODE_CLASS_MAPPINGS = {
    "QwenSampler": QwenSampler,
    "QwenLoader": QwenLoader,
    "DeepSeekResponseParser": DeepSeekResponseParser
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenSampler": "Qwen Sampler",
    "QwenLoader": "Qwen Loader",
    "DeepSeekResponseParser": "DeepSeek Response Parser"
}
