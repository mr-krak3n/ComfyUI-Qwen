# ComfyUI-Qwen

This repository contains custom nodes for ComfyUI, designed to facilitate working with language models such as Qwen2.5 and DeepSeek. The custom nodes include the following:

## Nodes

### QwenLoader
Used to load the language model. It has been tested with:
- Qwen2.5-3B
- Qwen2.5-7B
- DeepSeek-R1-Distill-Qwen-7B

### QwenSampler
Used for actual generation tasks. It is highly recommended to use a high `max_tokens` value when working with distilled models from DeepSeek. If you encounter VRAM limitations, enable the `keep_model_loaded` option to offload the model to CPU when not in use.

### DeepSeekResponseParser
A parser for separating `<think>` and `</think>` tags from the resulting text.

## Installation and Model Setup

1. Install the dependencies from `requirements.txt`. 
   - **Note**: If you are not using `load_in_4bit` with the QwenLoader node, you do not need to install the `BitsAndBytes` package.

2. Place the models inside subfolders within the `models/LLM` directory.

## Tested Models

Here are some models that have been tested and worked well:

- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen2.5-7B-Instruct-Uncensored](https://huggingface.co/Orion-zhen/Qwen2.5-7B-Instruct-Uncensored)
- [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)

Additionally, you can try out my finetuned models trained with Flux prompts:

- [Qwen2.5-3B-Instruct-Flux](https://huggingface.co/mrkrak3n/Qwen2.5-3B-Instruct-Flux)
- [Qwen2.5-7B-Instruct-Uncensored-Flux](https://huggingface.co/mrkrak3n/Qwen2.5-7B-Instruct-Uncensored-Flux)
- [Qwen2.5-7B-Instruct-Flux](https://huggingface.co/mrkrak3n/Qwen2.5-7B-Instruct-Flux)

## License

Feel free to modify, share, or contribute!