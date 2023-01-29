# EzP2P - Prompt-to-prompt extention of Web UI

## What's this?

This is an extension for [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that adds a custom script, which let you to use [prompt-to-prompt](https://github.com/google/prompt-to-prompt).

## Examples

<pre>
Setting:
  Sampling method: DPM++ 2M Karras
  Sampling steps: 15
  Size: 512x512
  CFG Scale: 6.0
  Seed: 1

Original:
  Prompt:   close up of a cute girl holing a <mark>cat</mark> in flower garden, insanely excess frilled white dress, small tiara
  Negative: (low quality, worst quality:1.4)

Prompt-to-prompt:
  Prompt:   close up of a cute girl holing a <mark>dragon</mark> in flower garden, insanely excess frilled white dress, small tiara
  Negative: (low quality, worst quality:1.4)
</pre>

- Original output:

![original](./images/cat.png)

- P2P output:

![p2p](./images/p2p_dragon.png)
