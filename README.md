# Auto 1111 SDK/Python Client

Auto 1111 SDK is a light-weight Python library for generating images, upscaling images, and editing images with diffusion models. It is designed to be a modular, light-weight Python client that encapsulates all the main features of the [Automatic 1111 Stable Diffusion Web Ui](https://github.com/AUTOMATIC1111/stable-diffusion-webui). Auto 1111 SDK offers 3 main core features currently:

- Text-to-Image, Image-to-Image, Inpainting, and Outpainting pipelines. Our pipelines support the exact same parameters as the [Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui), so you can easily replicate creations from the Web UI on the SDK.
- Upscaling Pipelines that can run inference for any Esrgan or Real Esrgan upscaler in a few lines of code.
- An integration with Civit AI to directly download models from the website.

Join our [Discord!!](https://discord.gg/S7wRQqt6QV)

## Installation

We recommend installing Auto 1111 SDK in a virtual environment from PyPI. Right now, we do not have support for conda environments yet.

```bash
pip3 install auto1111sdk
```

## Quickstart

Generating images with Auto 1111 SDK is super easy. To run inference for Text-to-Image, Image-to-Image, Inpainting, Outpainting, or Stable Diffusion Upscale, we have 1 pipeline that can support all these operations. This saves a lot of RAM from having to create multiple pipeline objects with other solutions.

```python
from auto1111sdk import StableDiffusionPipeline

pipe = StableDiffusionPipeline("<Path to your local safetensors or checkpoint file>")

prompt = "a picture of a brown dog"
output = pipe.generate_txt2img(prompt = prompt, height = 1024, width = 768, steps = 10)

output[0].save("image.png")
```

## Documentation

We have more detailed examples/documentation of how you can use Auto 1111 SDK [here.](https://flush-ai.gitbook.io/automatic-1111-sdk/). 
For a detailed comparison between us and Huggingface diffusers, you can read [this.](https://flush-ai.gitbook.io/automatic-1111-sdk/auto-1111-sdk-vs-huggingface-diffusers).


## Features
- Original txt2img and img2img modes
- Real ESRGAN upscale and Esrgan Upscale (compatible with any pth file)
- Outpainting
- Inpainting
- Stable Diffusion Upscale
- Attention, specify parts of text that the model should pay more attention to
    - a man in a `((tuxedo))` - will pay more attention to tuxedo
    - a man in a `(tuxedo:1.21)` - alternative syntax
    - select text and press `Ctrl+Up` or `Ctrl+Down` (or `Command+Up` or `Command+Down` if you're on a MacOS) to automatically adjust attention to selected text (code contributed by anonymous user)
- Composable Diffusion: a way to use multiple prompts at once
    - separate prompts using uppercase AND
    - also supports weights for prompts: a cat :1.2 AND a dog AND a penguin :2.2
- Works with a variety of samplers
- Download models directly from Civit AI and RealEsrgan checkpoints

## Limitations

- No support for Stable Diffusion XL checkpoints on GPUs.
- No support Hires Fix and Refiner parameters for inference.
- No support for Lora's
- No support for Face restoration
- No support for Dreambooth training script.

We will be adding support for these features very soon (in the priority of their order). We also accept any contributions to work on these issues!

## Contributing

Auto1111 SDK is continuously evolving, and we appreciate community involvement. We welcome all forms of contributions - bug reports, feature requests, and code contributions.

Report bugs and request features by opening an issue on Github.
Contribute to the project by forking/cloning the repository and submitting a pull request with your changes.


## Credits
Licenses for borrowed code can be found in `Settings -> Licenses` screen, and also in `html/licenses.html` file.

- Automatic 1111 Stable Diffusion Web UI - https://github.com/AUTOMATIC1111/stable-diffusion-webui
- Stable Diffusion - https://github.com/Stability-AI/stablediffusion, https://github.com/CompVis/taming-transformers
- k-diffusion - https://github.com/crowsonkb/k-diffusion.git
- ESRGAN - https://github.com/xinntao/ESRGAN
- MiDaS - https://github.com/isl-org/MiDaS
- Ideas for optimizations - https://github.com/basujindal/stable-diffusion
- Cross Attention layer optimization - Doggettx - https://github.com/Doggettx/stable-diffusion, original idea for prompt editing.
- Cross Attention layer optimization - InvokeAI, lstein - https://github.com/invoke-ai/InvokeAI (originally http://github.com/lstein/stable-diffusion)
- Sub-quadratic Cross Attention layer optimization - Alex Birch (https://github.com/Birch-san/diffusers/pull/1), Amin Rezaei (https://github.com/AminRezaei0x443/memory-efficient-attention)
- Textual Inversion - Rinon Gal - https://github.com/rinongal/textual_inversion (we're not using his code, but we are using his ideas).
- Idea for SD upscale - https://github.com/jquesnelle/txt2imghd
- Noise generation for outpainting mk2 - https://github.com/parlance-zz/g-diffuser-bot
- CLIP interrogator idea and borrowing some code - https://github.com/pharmapsychotic/clip-interrogator
- Idea for Composable Diffusion - https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch
- xformers - https://github.com/facebookresearch/xformers
- Sampling in float32 precision from a float16 UNet - marunine for the idea, Birch-san for the example Diffusers implementation (https://github.com/Birch-san/diffusers-play/tree/92feee6)
