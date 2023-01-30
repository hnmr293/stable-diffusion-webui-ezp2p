import gradio as gr

import torch
from torch import nn, Tensor
from ldm.modules.attention import SpatialTransformer, BasicTransformerBlock, CrossAttention, MemoryEfficientCrossAttention # type: ignore
from modules.processing import process_images, fix_seed, Processed, StableDiffusionProcessing
from modules import scripts
from scripts.ezp2plib.sdhook import SDHook

NAME = "EzP2P"

def E(msg: str):
    return f"[{NAME}] {msg}"

class SDGlobalExit(RuntimeError):
    pass

class ContextExtractor(SDHook):
    
    context: Tensor|None
    
    def __init__(self, enabled):
        super().__init__(enabled)
        self.context = None
    
    def hook_unet(self, p: StableDiffusionProcessing, unet: nn.Module):
        # UNetModel.forward
        def hook(module, org, *args, **kwargs):
            self.context = kwargs.get("context", None)
            raise SDGlobalExit()
        self.hook_forward(unet, hook)

class AttnHooker(SDHook):
    
    def setup(self, p: StableDiffusionProcessing, replace_ctx: Tensor):
        if self.enabled:
            self.ctx = replace_ctx
        return super().setup(p)
    
    def hook_unet(self, p: StableDiffusionProcessing, unet: nn.Module):
        def hook(mod, fn, *args, **kwargs):
            result = fn(self.ctx)
            return result
        
        assert self.ctx is not None
        for n, d, block, attn1, attn2 in self.each_attn(unet):
            self.hook_forward(attn2.to_v, hook)
        
    def each_attn(self, unet: nn.Module):
        def each_unet_block(unet):
            for block in unet.input_blocks:
                yield block
            yield unet.middle_block
            for block in unet.output_blocks:
                yield block
            
        def each_transformer(unet_block):
            for block in unet_block.children():
                if isinstance(block, SpatialTransformer):
                    yield block
        
        def each_basic_block(trans):
            for block in trans.children():
                if isinstance(block, BasicTransformerBlock):
                    yield block
        
        for block in each_unet_block(unet):
            for n, trans in enumerate(each_transformer(block)):
                for depth, basic_block in enumerate(each_basic_block(trans.transformer_blocks)):
                    attn1: CrossAttention|MemoryEfficientCrossAttention
                    attn2: CrossAttention|MemoryEfficientCrossAttention
                    
                    attn1, attn2 = basic_block.attn1, basic_block.attn2
                    assert isinstance(attn1, CrossAttention) or isinstance(attn1, MemoryEfficientCrossAttention)
                    assert isinstance(attn2, CrossAttention) or isinstance(attn2, MemoryEfficientCrossAttention)
                    
                    yield n, depth, basic_block, attn1, attn2
        
class Script(scripts.Script):

    def __init__(self):
        super().__init__()
    
    def title(self):
        return NAME
    
    def show(self, is_img2img):
        return True
    
    def ui(self, is_img2img):
        mode = 'img2img' if is_img2img else 'txt2img'
        id = lambda x: f"ezp2p-{mode}-{x}"
        with gr.Group():
            with gr.Row():
                copy1 = gr.Button(value="\U0001F847 Copy Prompt", elem_id=id("copy1"))
                copy2 = gr.Button(value="\U0001F847 Copy Negative Prompt", elem_id=id("copy2"))
            
        with gr.Group():
            r = gr.Slider(minimum=-1.0, maximum=2.0, value=1.0, step=0.01, label="Strength")
            c = gr.Textbox(placeholder="Prompt", label="Prompt", lines=4, elem_id=id("prompt"))
            uc = gr.Textbox(placeholder="Negative Prompt", label="Negative Prompt", lines=4, elem_id=id("neg-prompt"))
        
        js1 = f"return gradioApp().querySelector('#{mode}_prompt textarea').value"
        js2 = f"return gradioApp().querySelector('#{mode}_neg_prompt textarea').value"
        copy1.click(_js=f"() => {{ {js1}; return []; }}", fn=None, inputs=[], outputs=[c])
        copy2.click(_js=f"() => {{ {js2}; return []; }}", fn=None, inputs=[], outputs=[uc])
        
        return [r, c, uc]

    def run(self,
            p: StableDiffusionProcessing,
            r: float,
            c: str,
            uc: str,
    ) -> Processed:
        
        fix_seed(p)
        
        print(E("1st step: extracting base context vectors..."))
        ctx1 = self.fetch_ctx(p, p.prompt, p.negative_prompt)
        
        print(E("2nd step: extracting replaced context vectors..."))
        ctx2 = self.fetch_ctx(p, c, uc)
        
        assert ctx1 is not None
        assert ctx2 is not None
        
        print(E("3rd step: generating images..."))
        proc = self.process_hook(p, r, ctx1, ctx2)
        for idx, text in enumerate(proc.infotexts):
            if len(text) != 0:
                text += "\n"
            text += f"{NAME}_prompt: {c}\n{NAME}_negative_prompt: {uc}\n{NAME}_ratio: {r}"
            proc.infotexts[idx] = text
        
        return proc
    
    def fetch_ctx(self, p: StableDiffusionProcessing, c: str, uc: str):
        o_c = p.prompt
        o_uc = p.negative_prompt
        o_cs = p.all_prompts
        o_ucs = p.all_negative_prompts
        
        p.prompt = c
        p.negative_prompt = uc
        if o_cs is not None:
            p.all_prompts = [c] * len(o_cs) # type: ignore
        if o_ucs is not None:
            p.all_negative_prompts = [uc] * len(o_ucs) # type: ignore
        
        hooker = ContextExtractor(True)
        hooker.setup(p)
        with hooker:
            try:
                process_images(p)
            except SDGlobalExit:
                pass
        
        p.prompt = o_c
        p.negative_prompt = o_uc
        p.all_prompts = o_cs
        p.all_negative_prompts = o_ucs
        
        return hooker.context
    
    def process_hook(
        self,
        p: StableDiffusionProcessing,
        r: float,
        ctx1: Tensor,
        ctx2: Tensor
    ):
        hooker = AttnHooker(True)
        ctx = torch.lerp(ctx1, ctx2, r)
        hooker.setup(p, ctx)
        with hooker:
            return process_images(p)
