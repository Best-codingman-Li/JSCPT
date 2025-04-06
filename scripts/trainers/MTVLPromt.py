import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerXX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DataManager
from dassl.data.data_manager import build_data_loader
from dassl.data.datasets import build_dataset
from dassl.data.samplers import build_sampler
from dassl.data.transforms import INTERPOLATION_MODES, build_transform
from tabulate import tabulate

from trainers.vision_benchmark.evaluation import construct_dataloader, construct_multitask_dataset

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class CMaskMLP(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual=True):
        super(CMaskMLP, self).__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        #self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.mask = nn.Parameter(torch.zeros(in_dim, requires_grad=True))
        #self.maskfc = nn.Linear(in_dim, in_dim)
        self.use_residual = use_residual
        # self.act_fn = nn.GELU()
        self.act_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mask = self.sigmoid(self.mask).unsqueeze(0)
        x = torch.mul(x, mask)
        residual = x

        x = self.fc1(x)
        x = self.act_fn(x)
        #x = self.fc2(x)

        if self.use_residual:
            x = x + residual
        return x


class MTVLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, task_nums):
        super().__init__()
        # DEFAULT is VPT
        n_cls = len(classnames)
        coop_n_ctx = cfg.TRAINER.MVLPT.COOP.N_CTX
        cocoop_n_ctx = cfg.TRAINER.MVLPT.COCOOP.N_CTX
        vpt_n_ctx = cfg.TRAINER.MVLPT.VPT.N_CTX

        coop_ctx_init = cfg.TRAINER.MVLPT.COOP.CTX_INIT
        cocoop_ctx_init = cfg.TRAINER.MVLPT.COCOOP.CTX_INIT
        vpt_ctx_init = cfg.TRAINER.MVLPT.VPT.CTX_INIT

        dtype = clip_model.dtype
        coop_ctx_dim = clip_model.ln_final.weight.shape[0]
        cocoop_ctx_dim = coop_ctx_dim
        vpt_ctx_dim = clip_model.visual.conv1.weight.shape[0]

        vis_dim = clip_model.visual.output_dim

        # HACK: this is for VisualTransformer model
        clip_patchsize = clip_model.visual.conv1.weight.shape[-1]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.task_nums = task_nums
        #self.coop_mask = [CMaskMLP(128, 128) for _ in range(self.task_nums)] #task-specific coop mask for text
        #self.vpt_mask = [CMaskMLP(128, 128) for _ in range(self.task_nums)] #task-specific vpt mask for image

        self.vpt_dropout = Dropout(cfg.TRAINER.MVLPT.VPT.DROPOUT )#multi
        self.vpt_deep = cfg.TRAINER.MVLPT.VPT.DEEP
        self.vpt_embeddings = None
        self.vpt_embeddings_deep = None
        if vpt_n_ctx != 0:
            if cfg.TRAINER.MVLPT.VPT.PROJECT > -1:
                vpt_dim = cfg.TRAINER.MVLPT.VPT.PROJECT
                self.vpt_proj = nn.Linear(
                    vpt_dim, vpt_ctx_dim).type(dtype)
                nn.init.kaiming_normal_(
                    self.vpt_proj.weight, a=0, mode='fan_out')
            else:
                vpt_dim = vpt_ctx_dim
                self.vpt_proj = nn.Identity()

            if vpt_ctx_init:
                # Don't support ctx init for MVLPT
                raise ValueError("CTX initiation scheme is not supported")
            else:
                # random initialization
                clip_patchsize = _pair(clip_patchsize)
                val = math.sqrt(6. / float(3 * reduce(mul, clip_patchsize, 1) + vpt_dim))  # noqa

                self.vpt_embeddings = nn.Parameter(torch.zeros(
                    1, vpt_n_ctx, vpt_dim, dtype=dtype))
                # xavier_uniform initialization
                nn.init.uniform_(self.vpt_embeddings.data, -val, val)

                if self.vpt_deep:  # noqa
                    self.vision_layers = len([k for k in clip_model.state_dict().keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])

                    self.vpt_embeddings_deep = nn.Parameter(torch.zeros(
                        self.vision_layers-1, vpt_n_ctx, vpt_dim, dtype=dtype))
                    # xavier_uniform initialization
                    nn.init.uniform_(self.vpt_embeddings_deep.data, -val, val)
                
                prompt_prefix = "a photo of a "

                print(f'VPT Initial context: "{prompt_prefix}"')
                print(f"VPT Number of context words (tokens): {vpt_n_ctx}")
            
        self.ctx = None
        if coop_n_ctx != 0:
            if coop_ctx_init:
                # use given words to initialize context vectors
                coop_ctx_init = coop_ctx_init.replace("_", " ")
                coop_n_ctx = len(coop_ctx_init.split(" "))
                prompt = clip.tokenize(coop_ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1 : 1 + coop_n_ctx, :]
                prompt_prefix = coop_ctx_init

            else:
                # random initialization
                if cfg.TRAINER.MVLPT.COOP.CSC:
                    print("Initializing class-specific contexts")
                    ctx_vectors = torch.empty(n_cls, coop_n_ctx, coop_ctx_dim, dtype=dtype)
                else:
                    print("Initializing a generic context")
                    ctx_vectors = torch.empty(coop_n_ctx, coop_ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * coop_n_ctx)

            print(f'COOP Initial context: "{prompt_prefix}"')
            print(f"COOP Number of context words (tokens): {coop_n_ctx}")

            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.mvlpt_proj = nn.Identity()
        if vpt_n_ctx != 0 and coop_n_ctx != 0:
            self.mvlpt_proj_ctx_dim = cfg.TRAINER.MVLPT.PROJECT_DIM
            
            if cfg.TRAINER.MVLPT.PROJECT_METHOD == 'identity':
                self.mvlpt_proj = nn.Identity()
            else:
                # match dimension
                self.mvlpt_proj_ctx_vpt_pre, self.mvlpt_proj_ctx_vpt_post = nn.Identity(), nn.Identity()
                self.mvlpt_proj_ctx_coop_pre, self.mvlpt_proj_ctx_coop_post = nn.Identity(), nn.Identity()

                if coop_ctx_dim != self.mvlpt_proj_ctx_dim :
                    self.mvlpt_proj_ctx_coop_pre = nn.Linear( coop_ctx_dim, self.mvlpt_proj_ctx_dim).type(dtype)#, dtype=dtype
                    self.mvlpt_proj_ctx_coop_post = nn.Linear( self.mvlpt_proj_ctx_dim , coop_ctx_dim).type(dtype)#, dtype=dtype 
                if vpt_ctx_dim != self.mvlpt_proj_ctx_dim:
                    self.mvlpt_proj_ctx_vpt_pre = nn.Linear( vpt_ctx_dim, self.mvlpt_proj_ctx_dim).type(dtype)#, dtype=dtype
                    self.mvlpt_proj_ctx_vpt_post = nn.Linear( self.mvlpt_proj_ctx_dim , vpt_ctx_dim).type(dtype)#, dtype=dtype

                if cfg.TRAINER.MVLPT.PROJECT_METHOD == 'mlp':
                    self.mvlpt_proj = nn.GeLU()
                    
                elif cfg.TRAINER.MVLPT.PROJECT_METHOD == 'transformer':
                    from clip.model import Transformer
                    self.mvlpt_proj = Transformer(width=self.mvlpt_proj_ctx_dim, layers=1, heads=1)
                    # for n, m in self.MVLPT_proj.named_modules():
                    #     m.type(dtype)

        self.cocoop_ctx = None
        if cocoop_n_ctx != 0:
            if cocoop_ctx_init:
                # use given words to initialize context vectors
                cocoop_ctx_init = cocoop_ctx_init.replace("_", " ")
                cocoop_n_ctx = len(cocoop_ctx_init.split(" "))
                prompt = clip.tokenize(cocoop_ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1 : 1 + cocoop_n_ctx, :]
                prompt_prefix = cocoop_ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(cocoop_n_ctx, cocoop_ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * cocoop_n_ctx)

            print(f'Initial context: "{prompt_prefix}"')
            print(f"Number of context words (tokens): {cocoop_n_ctx}")

            self.cocoop_ctx = nn.Parameter(ctx_vectors)

            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(vis_dim // 16, cocoop_ctx_dim))
            ]))
            
            if cfg.TRAINER.MVLPT.COCOOP.PREC == "fp16":
                self.meta_net.half()


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
    
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        if cfg.TRAINER.CUT_CONTEXTLEN:
            sot_token = _tokenizer.encoder["<|startoftext|>"]
            eot_token = _tokenizer.encoder["<|endoftext|>"]
            max_length = min(clip_model.context_length, max([len([sot_token] + _tokenizer.encode(p) + [eot_token]) for p in prompts]))
        else:
            max_length = clip_model.context_length
        print("Current Context Length is: ", max_length)

        # exit()
        tokenized_prompts = torch.cat([clip.tokenize(p, context_length=max_length) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if cocoop_n_ctx != 0:
            self.register_buffer("token_suffix", embedding[:, 1 + cocoop_n_ctx :, :])  # CLS, EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + coop_n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.vpt_n_ctx = vpt_n_ctx
        self.coop_n_ctx = coop_n_ctx
        self.cocoop_n_ctx = cocoop_n_ctx

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward_cocoop(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.cocoop_ctx                     # (n_ctx, ctx_dim)
        if ctx is None:
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            return prompts
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts

    def forward_mvlpt_proj(self, dtype=torch.float):
        if self.coop_n_ctx == 0 or isinstance(self.mvlpt_proj, nn.Identity) or self.vpt_n_ctx == 0:
            return self.ctx, self.vpt_embeddings, self.vpt_embeddings_deep
        
        # print('vpt', self.vpt_embeddings.dtype, 'vpt_proj', self.MVLPT_proj_ctx_vpt_pre.weight.dtype)
        # print('coop_emb', self.vpt_embeddings.dtype, 'coop_emb_proj', self.MVLPT_proj_ctx_vpt_pre.weight.dtype)

        vpt_emb = self.vpt_embeddings # 1*vpt_n_ctx*vpt_ctx_dim (1, 16, 768)
        if self.vpt_deep:
            vpt_emb = torch.cat([ vpt_emb, self.vpt_embeddings_deep ], dim=0) # vision_layers*vpt_n_ctx*vpt_ctx_dim (11, 16, 768)

        vpt_ctx_dim = vpt_emb.shape[-1]
        vpt_emb = vpt_emb.reshape(1, -1, vpt_ctx_dim)

        coop_emb = self.ctx # n_ctx, ctx_dim or n_cls, n_ctx, ctx_dim 
        coop_ctx_dim = self.ctx.shape[-1]

        if coop_emb.dim() == 2:
            coop_emb = coop_emb.unsqueeze(0)
        coop_emb = coop_emb.reshape(1, -1, coop_ctx_dim)

        coop_emb_n_ctx = coop_emb.shape[1]

        # match dimension
        coop_emb = self.mvlpt_proj_ctx_coop_pre( coop_emb ) #[1, 16, 128]
        vpt_emb = self.mvlpt_proj_ctx_vpt_pre( vpt_emb ) #[1, 192, 128]

        '''
        c_coop_emb = self.coop_mask[task_index](coop_emb.squeeze(0))#task-specific masked coop prompts
        c_coop_emb = c_coop_emb.unsqueeze(0)
        c_vpt_emb = self.vpt_mask[task_index](vpt_emb.squeeze(0))#task-specific masked vpt prompts
        c_vpt_emb = c_vpt_emb.unsqueeze(0)
        '''

        mvlpt_emb = torch.cat([ coop_emb, vpt_emb ], dim=1) #concatenate the textual and visual prompt token embedding. Here, can add shared and specific embedings
        #c_mvlpt_emb = torch.cat([ c_coop_emb, c_vpt_emb ], dim=1)

        # print('mvlpt_emb', mvlpt_emb.dtype, 'mvlpt_emb_proj', self.MVLPT_proj.resblocks[0].attn.in_proj_weight.dtype)
        mvlpt_emb = self.mvlpt_proj( mvlpt_emb.float() )
        mvlpt_emb = mvlpt_emb.type(dtype)

        coop_emb, vpt_emb = mvlpt_emb[:, :coop_emb_n_ctx, :], mvlpt_emb[:, coop_emb_n_ctx:, :]
        #c_coop_emb, c_vpt_emb = c_mvlpt_emb[:, :coop_emb_n_ctx, :], c_mvlpt_emb[:, coop_emb_n_ctx:, :]
        
        coop_emb = self.mvlpt_proj_ctx_coop_post(coop_emb).reshape(-1, self.coop_n_ctx, coop_ctx_dim).squeeze(0)#multi
        vpt_emb = self.mvlpt_proj_ctx_vpt_post(vpt_emb).reshape(-1, self.vpt_n_ctx, vpt_ctx_dim)#multi
        vpt_emb_deep = None if vpt_emb.shape[0] == 1 else vpt_emb[1:, :, :]
        vpt_emb = vpt_emb[0, :, :].unsqueeze(0)

        return coop_emb, vpt_emb, vpt_emb_deep

    def forward_vpt(self, x, vpt_embeddings=None):
        B = x.shape[0] # (batch_size, 1 + n_patches, hidden_dim)

        if vpt_embeddings is None:
            if self.vpt_embeddings is None:
                return x
            vpt_embeddings = self.vpt_embeddings
        
        ctx = self.vpt_dropout(self.vpt_proj(vpt_embeddings).expand(B, -1, -1)).to(x.dtype)
        prefix = x[:, :1, :]
        suffix = x[:, 1:, :]

        prompts = torch.cat(
            [
                prefix,  # (B, 1, dim)
                ctx,     # (B, n_ctx, dim)
                suffix,  # (B, n_patches, dim)
            ],
            dim=1,
        )

        return prompts

    def forward_coop(self, ctx=None):
        if ctx is None:
            ctx = self.ctx
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        if ctx is None:
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            return prompts
        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.coop_n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts



class ImageEncoder(nn.Module):

    def __init__(self, clip_model, mvlpt_model):
        super().__init__()
        # HACK: Assume all is vision transformer
        self.visual = clip_model.visual
        self.mvlpt_model = mvlpt_model

    def forward(self, x: torch.Tensor, vpt_embeddings=None, vpt_embeddings_deep=None):

        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)#[64, 197, 768]

        #print("self.visual.ln_pre(x).size()============", x.size()) #[64, 197, 768]
        B = x.shape[0]

        x = self.mvlpt_model.forward_vpt(x, vpt_embeddings)
        x = x.permute(1, 0, 2)  # NLD -> LND

        if self.mvlpt_model.vpt_deep and (vpt_embeddings_deep is not None or self.mvlpt_model.vpt_embeddings_deep is not None):
            if vpt_embeddings_deep is None:
                vpt_embeddings_deep = self.mvlpt_model.vpt_embeddings_deep
            for layer_idx in range(self.visual.transformer.layers):
                layer = self.visual.transformer.resblocks[layer_idx]
                
                if layer_idx == 0:
                    x = layer(x)
                elif layer_idx <= vpt_embeddings_deep.shape[0]:
                    vpt_emb_deep = self.mvlpt_model.vpt_dropout(self.mvlpt_model.vpt_proj(
                        vpt_embeddings_deep[layer_idx-1]).expand(B, -1, -1)).to(x.dtype)

                    vpt_emb_deep = vpt_emb_deep.permute(1, 0, 2)  # NLD -> LND
                    x = torch.cat((
                        x[:1, :, :],
                        vpt_emb_deep,
                        x[(1+self.mvlpt_model.vpt_n_ctx):, :, :]
                    ), dim=0)
                    x = layer(x)
        else:
            x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x[:, 0, :])#[64, 768]
        #print("self.visual.ln_post(x[:, 0, :]).size()============", x.size())#torch.Size([64, 768]
        if self.visual.proj is not None:
            x = x @ self.visual.proj

        return x

class TextEncoder(nn.Module):
    def __init__(self, clip_model, cfg=None):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.cfg = cfg

    def forward(self, prompts, tokenized_prompts):
        if not self.cfg.TRAINER.CUT_CONTEXTLEN:
            x = prompts + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        else:
            x = prompts + self.positional_embedding.type(self.dtype)[:prompts.shape[1], :]
            x = x.permute(1, 0, 2)  # NLD -> LND
            
            for block in self.transformer.resblocks:
                if block.attn_mask.shape[0] != x.shape[0]:
                    block.attn_mask = block.attn_mask[:x.shape[0], :x.shape[0]]
            # x = self.transformer(x)
            from torch.utils.checkpoint import checkpoint_sequential
            act_chunk_size = min(self.cfg.TRAINER.ACT_CKPT, len(self.transformer.resblocks))
            x = checkpoint_sequential(self.transformer.resblocks, act_chunk_size, x) 
            x = x.permute(1, 0, 2)  # LND -> NLD
        
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

from torch.nn import Dropout
import math
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair

class MultitaskVLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # DEFAULT is VPT
        n_cls = len(classnames)
        coop_n_ctx = cfg.TRAINER.MVLPT.COOP.N_CTX
        cocoop_n_ctx = cfg.TRAINER.MVLPT.COCOOP.N_CTX
        vpt_n_ctx = cfg.TRAINER.MVLPT.VPT.N_CTX

        coop_ctx_init = cfg.TRAINER.MVLPT.COOP.CTX_INIT
        cocoop_ctx_init = cfg.TRAINER.MVLPT.COCOOP.CTX_INIT
        vpt_ctx_init = cfg.TRAINER.MVLPT.VPT.CTX_INIT

        dtype = clip_model.dtype
        coop_ctx_dim = clip_model.ln_final.weight.shape[0]
        cocoop_ctx_dim = coop_ctx_dim
        vpt_ctx_dim = clip_model.visual.conv1.weight.shape[0]

        vis_dim = clip_model.visual.output_dim

        # HACK: this is for VisualTransformer model
        clip_patchsize = clip_model.visual.conv1.weight.shape[-1]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        #self.coop_mask = MaskMLP(128, 128, 128)
        #self.vpt_mask = MaskMLP(128, 128, 128)

        self.vpt_dropout = Dropout(cfg.TRAINER.MVLPT.VPT.DROPOUT )
        self.vpt_deep = cfg.TRAINER.MVLPT.VPT.DEEP
        self.vpt_embeddings = None
        self.vpt_embeddings_deep = None
        if vpt_n_ctx != 0:
            if cfg.TRAINER.MVLPT.VPT.PROJECT > -1:
                vpt_dim = cfg.TRAINER.MVLPT.VPT.PROJECT
                self.vpt_proj = nn.Linear(
                    vpt_dim, vpt_ctx_dim).type(dtype)
                nn.init.kaiming_normal_(
                    self.vpt_proj.weight, a=0, mode='fan_out')
            else:
                vpt_dim = vpt_ctx_dim
                self.vpt_proj = nn.Identity()

            if vpt_ctx_init:
                # Don't support ctx init for MVLPT
                raise ValueError("CTX initiation scheme is not supported")
            else:
                # random initialization
                clip_patchsize = _pair(clip_patchsize)
                val = math.sqrt(6. / float(3 * reduce(mul, clip_patchsize, 1) + vpt_dim))  # noqa

                self.vpt_embeddings = nn.Parameter(torch.zeros(
                    1, vpt_n_ctx, vpt_dim, dtype=dtype))
                # xavier_uniform initialization
                nn.init.uniform_(self.vpt_embeddings.data, -val, val)

                if self.vpt_deep:  # noqa
                    self.vision_layers = len([k for k in clip_model.state_dict().keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])

                    self.vpt_embeddings_deep = nn.Parameter(torch.zeros(
                        self.vision_layers-1, vpt_n_ctx, vpt_dim, dtype=dtype))
                    # xavier_uniform initialization
                    nn.init.uniform_(self.vpt_embeddings_deep.data, -val, val)
                
                prompt_prefix = "a photo of a "

                print(f'VPT Initial context: "{prompt_prefix}"')
                print(f"VPT Number of context words (tokens): {vpt_n_ctx}")
            
        self.ctx = None
        if coop_n_ctx != 0:
            if coop_ctx_init:
                # use given words to initialize context vectors
                coop_ctx_init = coop_ctx_init.replace("_", " ")
                coop_n_ctx = len(coop_ctx_init.split(" "))
                prompt = clip.tokenize(coop_ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1 : 1 + coop_n_ctx, :]
                prompt_prefix = coop_ctx_init

            else:
                # random initialization
                if cfg.TRAINER.MVLPT.COOP.CSC:
                    print("Initializing class-specific contexts")
                    ctx_vectors = torch.empty(n_cls, coop_n_ctx, coop_ctx_dim, dtype=dtype)
                else:
                    print("Initializing a generic context")
                    ctx_vectors = torch.empty(coop_n_ctx, coop_ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * coop_n_ctx)

            print(f'COOP Initial context: "{prompt_prefix}"')
            print(f"COOP Number of context words (tokens): {coop_n_ctx}")

            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.mvlpt_proj = nn.Identity()
        if vpt_n_ctx != 0 and coop_n_ctx != 0:
            self.mvlpt_proj_ctx_dim = cfg.TRAINER.MVLPT.PROJECT_DIM
            
            if cfg.TRAINER.MVLPT.PROJECT_METHOD == 'identity':
                self.mvlpt_proj = nn.Identity()
            else:
                # match dimension
                self.mvlpt_proj_ctx_vpt_pre, self.mvlpt_proj_ctx_vpt_post = nn.Identity(), nn.Identity()
                self.mvlpt_proj_ctx_coop_pre, self.mvlpt_proj_ctx_coop_post = nn.Identity(), nn.Identity()

                if coop_ctx_dim != self.mvlpt_proj_ctx_dim :
                    self.mvlpt_proj_ctx_coop_pre = nn.Linear( coop_ctx_dim, self.mvlpt_proj_ctx_dim).type(dtype)#, dtype=dtype
                    self.mvlpt_proj_ctx_coop_post = nn.Linear( self.mvlpt_proj_ctx_dim , coop_ctx_dim).type(dtype)#, dtype=dtype 
                if vpt_ctx_dim != self.mvlpt_proj_ctx_dim:
                    self.mvlpt_proj_ctx_vpt_pre = nn.Linear( vpt_ctx_dim, self.mvlpt_proj_ctx_dim).type(dtype)#, dtype=dtype
                    self.mvlpt_proj_ctx_vpt_post = nn.Linear( self.mvlpt_proj_ctx_dim , vpt_ctx_dim).type(dtype)#, dtype=dtype

                if cfg.TRAINER.MVLPT.PROJECT_METHOD == 'mlp':
                    self.mvlpt_proj = nn.GeLU()
                    
                elif cfg.TRAINER.MVLPT.PROJECT_METHOD == 'transformer':
                    from clip.model import Transformer
                    self.mvlpt_proj = Transformer(width=self.mvlpt_proj_ctx_dim, layers=1, heads=1)
                    # for n, m in self.MVLPT_proj.named_modules():
                    #     m.type(dtype)

        self.cocoop_ctx = None
        if cocoop_n_ctx != 0:
            if cocoop_ctx_init:
                # use given words to initialize context vectors
                cocoop_ctx_init = cocoop_ctx_init.replace("_", " ")
                cocoop_n_ctx = len(cocoop_ctx_init.split(" "))
                prompt = clip.tokenize(cocoop_ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1 : 1 + cocoop_n_ctx, :]
                prompt_prefix = cocoop_ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(cocoop_n_ctx, cocoop_ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * cocoop_n_ctx)

            print(f'Initial context: "{prompt_prefix}"')
            print(f"Number of context words (tokens): {cocoop_n_ctx}")

            self.cocoop_ctx = nn.Parameter(ctx_vectors)

            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(vis_dim // 16, cocoop_ctx_dim))
            ]))
            
            if cfg.TRAINER.MVLPT.COCOOP.PREC == "fp16":
                self.meta_net.half()


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
    
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        if cfg.TRAINER.CUT_CONTEXTLEN:
            sot_token = _tokenizer.encoder["<|startoftext|>"]
            eot_token = _tokenizer.encoder["<|endoftext|>"]
            max_length = min(clip_model.context_length, max([len([sot_token] + _tokenizer.encode(p) + [eot_token]) for p in prompts]))
        else:
            max_length = clip_model.context_length
        print("Current Context Length is: ", max_length)

        # exit()
        tokenized_prompts = torch.cat([clip.tokenize(p, context_length=max_length) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if cocoop_n_ctx != 0:
            self.register_buffer("token_suffix", embedding[:, 1 + cocoop_n_ctx :, :])  # CLS, EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + coop_n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.vpt_n_ctx = vpt_n_ctx
        self.coop_n_ctx = coop_n_ctx
        self.cocoop_n_ctx = cocoop_n_ctx

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward_cocoop(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.cocoop_ctx                     # (n_ctx, ctx_dim)
        if ctx is None:
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            return prompts
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts

    def forward_mvlpt_proj(self, dtype=torch.float):
        if self.coop_n_ctx == 0 or isinstance(self.mvlpt_proj, nn.Identity) or self.vpt_n_ctx == 0:
            return self.ctx, self.vpt_embeddings, self.vpt_embeddings_deep
        
        # print('vpt', self.vpt_embeddings.dtype, 'vpt_proj', self.MVLPT_proj_ctx_vpt_pre.weight.dtype)
        # print('coop_emb', self.vpt_embeddings.dtype, 'coop_emb_proj', self.MVLPT_proj_ctx_vpt_pre.weight.dtype)

        vpt_emb = self.vpt_embeddings # 1*vpt_n_ctx*vpt_ctx_dim (1, 16, 768)
        if self.vpt_deep:
            vpt_emb = torch.cat([ vpt_emb, self.vpt_embeddings_deep ], dim=0) # vision_layers*vpt_n_ctx*vpt_ctx_dim (11, 16, 768)

        vpt_ctx_dim = vpt_emb.shape[-1]
        vpt_emb = vpt_emb.reshape(1, -1, vpt_ctx_dim)

        coop_emb = self.ctx # n_ctx, ctx_dim or n_cls, n_ctx, ctx_dim 
        coop_ctx_dim = self.ctx.shape[-1]

        if coop_emb.dim() == 2:
            coop_emb = coop_emb.unsqueeze(0)
        coop_emb = coop_emb.reshape(1, -1, coop_ctx_dim)

        coop_emb_n_ctx = coop_emb.shape[1]

        # match dimension
        coop_emb = self.mvlpt_proj_ctx_coop_pre( coop_emb ) #[1, 16, 128]
        vpt_emb = self.mvlpt_proj_ctx_vpt_pre( vpt_emb ) #[1, 192, 128]

        '''
        coop_emb, _ = self.coop_mask(coop_emb.squeeze(0))
        coop_emb = coop_emb.unsqueeze(0)
        vpt_emb, _ = self.coop_mask(vpt_emb.squeeze(0))
        vpt_emb = vpt_emb.unsqueeze(0)
        '''

        mvlpt_emb = torch.cat([ coop_emb, vpt_emb ], dim=1) #concatenate the textual and visual prompt token embedding. Here, can add shared and specific embedings

        # print('mvlpt_emb', mvlpt_emb.dtype, 'mvlpt_emb_proj', self.MVLPT_proj.resblocks[0].attn.in_proj_weight.dtype)
        mvlpt_emb = self.mvlpt_proj( mvlpt_emb.float() )
        mvlpt_emb = mvlpt_emb.type(dtype)
        coop_emb, vpt_emb = mvlpt_emb[:, :coop_emb_n_ctx, :], mvlpt_emb[:, coop_emb_n_ctx:, :]
        
        coop_emb = self.mvlpt_proj_ctx_coop_post(coop_emb).reshape(-1, self.coop_n_ctx, coop_ctx_dim).squeeze(0)
        vpt_emb = self.mvlpt_proj_ctx_vpt_post(vpt_emb).reshape(-1, self.vpt_n_ctx, vpt_ctx_dim)
        vpt_emb_deep = None if vpt_emb.shape[0] == 1 else vpt_emb[1:, :, :]
        vpt_emb = vpt_emb[0, :, :].unsqueeze(0)
        return coop_emb, vpt_emb, vpt_emb_deep

    def forward_vpt(self, x, vpt_embeddings=None):
        B = x.shape[0] # (batch_size, 1 + n_patches, hidden_dim)

        if vpt_embeddings is None:
            if self.vpt_embeddings is None:
                return x
            vpt_embeddings = self.vpt_embeddings
        
        ctx = self.vpt_dropout(self.vpt_proj(vpt_embeddings).expand(B, -1, -1)).to(x.dtype)
        prefix = x[:, :1, :]
        suffix = x[:, 1:, :]

        prompts = torch.cat(
            [
                prefix,  # (B, 1, dim)
                ctx,     # (B, n_ctx, dim)
                suffix,  # (B, n_patches, dim)
            ],
            dim=1,
        )

        return prompts

    def forward_coop(self, ctx=None):
        if ctx is None:
            ctx = self.ctx
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        if ctx is None:
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            return prompts
        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.coop_n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts




class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, dm=None):
        super().__init__()

        self.task_nums = len(dm._task_names)
        self.dtype = clip_model.dtype
        self.prompt_learner = MTVLPromptLearner(cfg, classnames, clip_model, self.task_nums)
        self.shared_tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder = ImageEncoder(clip_model, self.prompt_learner)
        self.text_encoder = TextEncoder(clip_model, cfg)
        self.logit_scale = clip_model.logit_scale
        
        
        self.vpt_mask = [CMaskMLP(768, 768).to("cuda") for _ in range(self.task_nums)]
        self.vpt_deep_mask = [CMaskMLP(768, 768).to("cuda") for _ in range(self.task_nums)]
        self.coop_mask = [CMaskMLP(512, 512).to("cuda") for _ in range(self.task_nums)]

        self.coop_n_ctx = cfg.TRAINER.MVLPT.COOP.N_CTX
        self.cocoop_n_ctx = cfg.TRAINER.MVLPT.COCOOP.N_CTX
        self.vpt_n_ctx = cfg.TRAINER.MVLPT.VPT.N_CTX

        #self.coop_linear = nn.Linear(1024, 512).type(self.dtype)
        #self.vpt_linear = nn.Linear(768*2, 768).type(self.dtype)
        #self.vpt_deep_linear = nn.Linear(768*self.vpt_n_ctx*2, 768*self.vpt_n_ctx).type(self.dtype)

        self.multi_task_label_pertask = cfg.DATASET.MULTITASK_LABEL_PERTASK
        if self.multi_task_label_pertask:
            self.class_index_pertask_start = torch.arange(dm._num_classes)
            self.class_index_pertask_end = torch.arange(dm._num_classes)
            start_index = 0

            #the num of tasks equals to the num of datasets, current num of tasks is 3, and the num of class is 535
            for class_index, task in enumerate(dm._task_names):
                class_num = len(dm._labelmap[task])
                self.class_index_pertask_start[class_index] = start_index
                start_index += class_num
                self.class_index_pertask_end[class_index] = start_index
            self.index = torch.arange(dm._num_classes).unsqueeze(0)

    def CustomCLIP_Logits(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def forward(self, image, task=None, train=True):
        shared_coop_emb, shared_vpt_emb, shared_vpt_emb_deep = self.prompt_learner.forward_mvlpt_proj(self.dtype) #vpt_emb.size()=[1, 16, 768];vpt_emb_deep.size()=[11, 16, 768];coop_emb.size()=[16,512]
        #shared_coop_emb, shared_vpt_emb, shared_vpt_emb_deep = shared_coop_emb.to('cuda'), shared_vpt_emb.to('cuda'), shared_vpt_emb_deep.to('cuda')

        coop_emb = self.coop_mask[task](shared_coop_emb)

        vpt_emb = self.vpt_mask[task](shared_vpt_emb.squeeze(0))
        vpt_emb = vpt_emb.unsqueeze(0)

        vpt_emb_deep = self.vpt_deep_mask[task](shared_vpt_emb_deep.reshape(-1, 768))
        vpt_emb_deep = vpt_emb_deep.reshape(-1, self.vpt_n_ctx, 768)

        '''
        coop_emb_ = torch.cat((shared_coop_emb.to('cuda'), task_sp_coop_emb.to('cuda')), dim=0).reshape(self.coop_n_ctx, -1)
        coop_emb = self.coop_linear(coop_emb_)

        vpt_emb_ = torch.cat((shared_vpt_emb.to('cuda'), task_sp_vpt_emb.to('cuda') ), dim=1).reshape(self.vpt_n_ctx, -1)
        vpt_emb = self.vpt_linear(vpt_emb_).unsqueeze(0)

        vpt_emb_deep_ = torch.cat((shared_vpt_emb_deep.to('cuda'), task_sp_vpt_emb_deep.to('cuda')), dim=1).reshape(11, -1)
        vpt_emb_deep = self.vpt_deep_linear(vpt_emb_deep_).reshape(11, self.vpt_n_ctx, -1)
        '''
        image_features = self.image_encoder(image.type(self.dtype), vpt_emb, vpt_emb_deep) #torch.Size([256, 512]
        #print("image_features.type()", image_features.type())
        if self.prompt_learner.cocoop_ctx == None:

            prompts = self.prompt_learner.forward_coop(coop_emb).type(self.dtype)#torch.Size([535, 25, 512])
            #print("prompts.type()", prompts.type())
            tokenized_prompts = self.shared_tokenized_prompts.type(self.dtype)#([535, 25])
            #print("tokenized_prompts.type()", tokenized_prompts.type())

            text_features = self.text_encoder(prompts, tokenized_prompts)#torch.Size([535, 512])
            logits = self.CustomCLIP_Logits(image_features, text_features)

            if train:

                shared_image_features = self.image_encoder(image.type(self.dtype), shared_vpt_emb, shared_vpt_emb_deep)#torch.Size([256, 512]
                shared_prompts = self.prompt_learner.forward_coop(shared_coop_emb).type(self.dtype)#torch.Size([535, 25, 512])
                #tokenized_prompts = self.shared_tokenized_prompts#([535, 25])
                shared_text_features = self.text_encoder(shared_prompts, tokenized_prompts)#torch.Size([535, 512])
                shared_logits = self.CustomCLIP_Logits(shared_image_features, shared_text_features)
            
        else:

            tokenized_prompts = self.tokenized_prompts
            logit_scale = self.logit_scale.exp()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            prompts = self.prompt_learner.forward_cocoop(image_features)
            
            logits = []
            for pts_i, imf_i in zip(prompts, image_features):
                text_features = self.text_encoder(pts_i, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                l_i = logit_scale * imf_i @ text_features.t()
                logits.append(l_i)
            logits = torch.stack(logits)

        if self.multi_task_label_pertask:
            print("----------Here we perform prompt selection----------")
            # Here we perform prompt selection
            domain_start_indexs = self.class_index_pertask_start[task].unsqueeze(-1)
            domain_end_indexs = self.class_index_pertask_end[task].unsqueeze(-1)
            # print(domain_start_indexs.shape, domain_end_indexs.shape, logits.shape)
            select_index = self.index.repeat(logits.shape[0], 1)
            select_index = (select_index >= domain_start_indexs).float() * (select_index < domain_end_indexs).float()
            # exit()
            logits = logits * select_index.to(logits.device)

        if train:
            return logits, shared_logits
        else:
            return logits


class MVLPTCOOPDataManager(DataManager):

    def __init__(self, cfg, custom_tfm_train=None, custom_tfm_test=None, dataset_wrapper=None):
        # Load dataset
        label_offset = 0
        self.num_classes_list = []
        self.classnames_list = []
        self.lab2cname_list = {}
        self.dataset = None
        self._task_names = cfg.DATASET.DATASET.split(',')
        self._id2task = {}
        self._task_class_idx = {}
        
        train_loader_x, train_loader_u, val_loader, test_loader = {}, {}, {}, {}

        for domain, dataset_name in enumerate(self._task_names):
            cfg.defrost()
            cfg.DATASET.NAME = dataset_name
            cfg.freeze()
            self._id2task[domain] = dataset_name
            dataset = build_dataset(cfg)
            self.num_classes_list.append(dataset._num_classes)
            self.classnames_list += dataset._classnames
            new_lab2cname_dict = {}
            for key, value in dataset._lab2cname.items():
                new_lab2cname_dict[key+label_offset] = value
            self.lab2cname_list.update(new_lab2cname_dict)
            for i in range(len(dataset._train_x)):
                dataset._train_x[i]._label += label_offset
                dataset._train_x[i]._domain = domain
            
            if dataset._train_u:
                for i in range(len(dataset._train_u)):
                    dataset._train_u[i]._label += label_offset
                    dataset._train_u[i]._domain = domain
                if self.dataset is not None:
                    self.dataset._train_u = self.dataset._train_u + dataset._train_u
            if dataset._val:
                for i in range(len(dataset._val)):
                    dataset._val[i]._label += label_offset
                    dataset._val[i]._domain = domain

            for i in range(len(dataset._test)):
                dataset._test[i]._label += label_offset
                dataset._test[i]._domain = domain
            
            if self.dataset is not None:
                self.dataset._train_x = self.dataset._train_x + dataset._train_x
                self.dataset._val = self.dataset.val + dataset.val
                self.dataset._test = self.dataset.test + dataset.test

            #print(dataset._train_u is None, dataset._val is None)
            if self.dataset is None:
                self.dataset = dataset

            self._task_class_idx[dataset_name] = ( label_offset, label_offset + dataset._num_classes )
            label_offset += dataset._num_classes

            #the last lines in for loop are added
            # Build transform
            if custom_tfm_train is None:
                tfm_train = build_transform(cfg, is_train=True)
            else:
                print("* Using custom transform for training")
                tfm_train = custom_tfm_train

            if custom_tfm_test is None:
                tfm_test = build_transform(cfg, is_train=False)
            else:
                print("* Using custom transform for testing")
                tfm_test = custom_tfm_test

            # Build train_loader_x
            #cfg.DATALOADER.TRAIN_X.SAMPLER,
            train_loader_x[domain] = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.train_x,
                batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

            # Build train_loader_u
            #cfg.DATALOADER.TRAIN_X.SAMPLER,
            train_loader_u[domain] = None
            if dataset.train_u:
                sampler_type_ = cfg.DATALOADER.TEST.SAMPLER #### replaced
                batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

                if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                    sampler_type_ = cfg.DATALOADER.TEST.SAMPLER ##### replaced
                    batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                    n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                    n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

                train_loader_u[domain] = build_data_loader(
                    cfg,
                    sampler_type=sampler_type_,
                    data_source=dataset.train_u,
                    batch_size=batch_size_,
                    n_domain=n_domain_,
                    n_ins=n_ins_,
                    tfm=tfm_train,
                    is_train=True,
                    dataset_wrapper=dataset_wrapper
                )
            # Build val_loader
            val_loader[domain] = None
            if dataset.val:
                val_loader[domain] = build_data_loader(
                    cfg,
                    sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                    data_source=dataset.val,
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    tfm=tfm_test,
                    is_train=False,
                    dataset_wrapper=dataset_wrapper
                )

            # Build test_loader
            test_loader[domain] = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.test,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        dataset = self.dataset
        dataset._classnames = self.classnames_list
        dataset._lab2cname = self.lab2cname_list
        dataset._num_classes = sum(self.num_classes_list)
        #print(self.num_classes_list, len(dataset._classnames), dataset._lab2cname, dataset._num_classes)

        '''
        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )
        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )
        '''
        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)


from trainers.vision_benchmark.datasets import class_map_metric, get_metric
import random
class MVLPTDataManager(DataManager):

    def __init__(self, cfg):
        # Load dataset
        train_loader_x, val_loader, test_loader, class_map, train_dataset = construct_dataloader(cfg)

        self._metric = get_metric(class_map_metric[cfg.DATASET.DATASET])
        self._metric_name = class_map_metric[cfg.DATASET.DATASET]

        # Attributes
        self._num_classes = len(class_map)
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = {}
        # random.seed(cfg.DATASET.RANDOM_SEED_SAMPLING)
        for key, value in enumerate(class_map):
            if isinstance(value, list):
                # value = random.choice(value)
                value = value[0]
            self._lab2cname[key] = value

        # Dataset and data-loaders
        # self.dataset.train_x = train_dataset
        self.train_loader_x = train_loader_x
        # self.train_loader_u = train_loader_u
        self.train_loader_u = None
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            pass
            # self.show_dataset_summary(cfg)

class MVLPTMTDataManager(DataManager):

    def __init__(self, cfg):
        # Load dataset
        train_loader_x, val_loader, test_loader, train_dataset, test_dataloader_by_task = construct_multitask_dataset(cfg)

        self._labelmap = train_dataset.labelmap
        self._task_names = train_dataset._task_names
        self._task2id = { v: k for k, v in enumerate(self._task_names) }
        self._id2task = { k: v for k, v in enumerate(self._task_names) }
        self._metric = { task: get_metric(class_map_metric[task]) for task in self._task_names }
        self._metric_name = { task: class_map_metric[task] for task in self._task_names }

        class_idx = 0
        self._task_class_idx = {}
        for task in self._task_names:
            class_num = len(self._labelmap[task])
            self._task_class_idx[task] = ( class_idx, class_idx + class_num )
            class_idx += class_num
        
        from trainers.vision_benchmark.datasets import class_map

        mt_class_map = dict()
        for task in self._labelmap:
            for label_idx, label in enumerate(class_map[task]):
                cnt = train_dataset._get_cid( label_idx, task)
                mt_class_map[cnt] = label
        
        # Attributes
        self._num_classes = len(mt_class_map)
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = {}
        
        # random.seed(cfg.DATASET.RANDOM_SEED_SAMPLING)
        for key, value in mt_class_map.items():
            if isinstance(value, list):
                value = value[0] #random.choice(value)
            self._lab2cname[key] = value

        # Dataset and data-loaders
        # self.dataset.train_x = train_dataset
        self.train_loader_x = train_loader_x
        # self.train_loader_u = train_loader_u
        self.train_loader_u = None
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            pass


@TRAINER_REGISTRY.register()
class MVLPT(TrainerXX):
    """Context Optimization (MVLPT).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MVLPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        if self.cfg.DATASET.COOP:
            classnames = self.dm.dataset.classnames
        else:
            classnames = self.dm.lab2cname.values()

        #print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.MVLPT.PREC == "fp32" or cfg.TRAINER.MVLPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, dm=self.dm)

        print("Turning off gradients in both the image and the text encoder")
        tune_param_name_set = ["prompt_learner", "vpt_mask", "vpt_deep_mask", "coop_mask"]
        shared_param_name_set = ["prompt_learner"]
        learner = [self.model.prompt_learner, self.model.vpt_mask, self.model.vpt_deep_mask, self.model.coop_mask]
        tune_params_set, shared_params_set = [], []

        for names, params in self.model.named_parameters():
            for t_name in tune_param_name_set:
                if t_name in names :
                    params.requires_grad_(True)
                    tune_params_set.append(params)
                    print(names, params.shape)
                    
        print(f"Tunable Param: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])/10**6}M, Original CLIP {sum([p.numel() for p in self.model.parameters() if not p.requires_grad])/10**6}M")
        
        if cfg.MODEL.INIT_WEIGHTS:
            #load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        #self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.optim = build_optimizer(self.model, cfg.OPTIM)

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        #self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)


        self.scaler = GradScaler() if cfg.TRAINER.MVLPT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    
    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        self.multi_task = self.cfg.DATASET.MULTITASK
        self.multi_task_label_pertask = self.cfg.DATASET.MULTITASK_LABEL_PERTASK

        if self.cfg.DATASET.COOP:
            dm = MVLPTCOOPDataManager(self.cfg)
        elif self.cfg.DATASET.MULTITASK:
            dm = MVLPTMTDataManager(self.cfg)
        else:
            dm = MVLPTDataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def forward_backward(self, batch, task_):
        #print("task-==-=", task_)

        loss_summary = {}
        image, label = self.parse_batch_train(batch)

        # HACK: for multi-label classification, either works
        if len(label.shape) > 1 and label.shape[-1] > 1:
            label = label.float()
            label /= label.sum(dim=-1, keepdim=True)
        
        prec = self.cfg.TRAINER.MVLPT.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image, task=task_)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            #print("image and textual features masked")
            output, shared_output = self.model(image, task=task_, train=True)
            #print(label.shape, output.shape, label.dtype, output.dtype, task_, label.sum(dim=-1))

            loss = F.cross_entropy(output, label)
            shared_loss = F.cross_entropy(shared_output, label)

            #self.model_backward_and_update(loss)

        # HACK: During training, we hack the eval of multi-label by selecting only one class
        if len(label.shape) > 1 and label.shape[-1] > 1:
            label = torch.argmax(label, dim=1)
        
        # result = self.dm._metric(label.squeeze().cpu().detach().numpy(), output.cpu().detach().numpy())

        loss_summary= {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            # "acc": result,
        }

        #if (self.batch_idx + 1) == self.num_batches:
            #self.update_lr()

        task_loss = loss / shared_loss
        causal_effect = torch.exp(shared_loss-loss)
        return loss_summary, task_loss, causal_effect

    def _prepare_dataloaders(self, dataloaders, task_name):

        loader = {}
        batch_num = []
        for task in task_name:
            loader[task] = [dataloaders[task], iter(dataloaders[task])]
            batch_num.append(len(dataloaders[task]))
        return loader, batch_num

    def _process_data(self, loader):
        try:
            batch = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            batch = next(loader[1])
        return batch

    def parse_batch_train(self, batch):
        if self.cfg.DATASET.COOP:
            inp_key, lab_key, task_key = 'img', 'label', 'domain'
        else:
            inp_key, lab_key, task_key = 0, 1, 3
        input = batch[inp_key]
        label = batch[lab_key]
        # print(label.shape, 'label', input.shape, 'input')
        tasks = None
        if self.multi_task:
            tasks = batch[task_key]
        # input = batch["img"]
        # label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        
        return input, label

    def parse_batch_test(self, batch):
        if self.cfg.DATASET.COOP:
            inp_key, lab_key, task_key = 'img', 'label', 'domain'
        else:
            inp_key, lab_key, task_key = 0, 1, 3
        input = batch[inp_key]
        label = batch[lab_key]
        tasks = None
        if self.multi_task:
            tasks = batch[task_key]
        # input = batch["img"]
        # label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def model_inference(self, input, task=None):
        return self.model(input, task=task, train=False)

    @torch.no_grad()
    def test(self, split=None):
        from tqdm import tqdm
        import copy 
        import numpy as np
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            print("-=-=-=val-=-=-=")
            data_loader = self.val_loader
        else:
            print("-=-=-=test-=-=-=")
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        self.evaluator_task = dict()

        self.elevator_evaluator = { 'y_pred': [], 'y_true': [] }
        #print("self.dm._task_names===", self.dm._task_names)#['Food101', 'OxfordPets', 'SUN397']
        if self.multi_task:
            if self.cfg.DATASET.COOP:
                #print("text self.cfg.DATASET.COOP")
                self.evaluator_task = { task: copy.deepcopy( self.evaluator ) for task in self.dm._task_names }
            else:
                self.evaluator_task = { task: copy.deepcopy( self.elevator_evaluator ) for task in self.dm._task_names }

        Task_test_acc = [0] * len(data_loader.keys())
        for task_ in data_loader.keys():
            
            #print("test test_batchs num", len(test_batchs))
            for batch_idx, batch in enumerate(tqdm(data_loader[task_])):
                image, label = self.parse_batch_train(batch)
                #print("test label", label)
                output = self.model_inference(image, task=task_)
                #print(output.size())#[bs, 535]
                #print(label.size())#[bs]

                Task_test_acc[task_] += (compute_accuracy(output, label)[0].item() / (batch_idx + 1))

                # HACK: make everything one-hot vector label!
                if self.cfg.DATASET.COOP:
                    self.evaluator.process(output, label)
                
                else:
                    self.elevator_evaluator['y_pred'].append( output.cpu().detach().numpy() )
                    self.elevator_evaluator['y_true'].append( label.cpu().detach().numpy() )

                tasks = [task_] * len(label)
                if tasks is not None:
                    for out, lab, task in zip(output, label, tasks):
                        #print("test per task", task)
                        #print("out", out)
                        #print("label", lab)
                        #task = self.dm._id2task[task.item()]
                        task = self.dm._id2task[task]
                        
                        if self.cfg.DATASET.COOP:
                            class_start, class_end = self.dm._task_class_idx[task]#[0, 101], [101, 138], [138, 535]
                            #print(class_start, class_end)
                            # Evaluate on the task-specific class
                            out = out[class_start:class_end]
                            lab -= class_start
                            self.evaluator_task[task].process(out.unsqueeze(0), lab.unsqueeze(0))
                        else:
                            self.evaluator_task[task]['y_pred'].append( [out.cpu().detach().numpy()] )
                            self.evaluator_task[task]['y_true'].append( [lab.cpu().detach().numpy()] )
        
        print("the test accuracy are:")
        for task_name in data_loader.keys():
            print(f"task :{self.dm._id2task[task_name]} | accuracy {Task_test_acc[task_name]:.3f}")

        results_overall = {}
        for task in self.evaluator_task:
            print(f"evaluate on the *{task}* !")
            if self.cfg.DATASET.COOP:
                results = self.evaluator_task[task].evaluate()
                results_overall[task] = results['accuracy']
            else:
                y_true = np.concatenate( self.evaluator_task[task]['y_true'] , axis=0)
                y_pred = np.concatenate( self.evaluator_task[task]['y_pred'] , axis=0)
                class_start, class_end = self.dm._task_class_idx[task]
                y_true = y_true[:, class_start:class_end]
                y_pred = y_pred[:, class_start:class_end]
                
                if self.dm._metric_name[task] == 'accuracy':
                    y_true = np.argmax(y_true, axis=-1)
                metric_result = self.dm._metric[task]( y_true, y_pred )
                results = { self.dm._metric_name[task]: metric_result }
                results_overall[ task ] = metric_result
            print( 'results', results )
            for k, v in results.items():
                tag = f"{split}/{task}/{k}"
                self.write_scalar(tag, v, self.epoch)
        
        print(f"Overall evaluation !")
        if self.multi_task:
            multi_task_evalkey = self.cfg.DATASET.MULTITASK_EVALKEY
            if multi_task_evalkey == 'average':
                results = {'average' : sum([v for k, v in results_overall.items()]) / len(results_overall)}
            else:
                assert multi_task_evalkey in results_overall
                results = {multi_task_evalkey : results_overall[multi_task_evalkey]}
                print(f"select {multi_task_evalkey} as the evaluation key")
        else:
            if not self.cfg.DATASET.COOP:
                y_true = np.concatenate( self.elevator_evaluator['y_true'] , axis=0)
                y_pred = np.concatenate( self.elevator_evaluator['y_pred'] , axis=0)
                results = { self.dm._metric_name: self.dm._metric( y_true, y_pred ) }
            else:
                results = self.evaluator.evaluate()
        print( 'results', results )
        for k, v in results.items():
            tag = f"/{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        
        return list(results.values())[0]


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            # issue 1 bug fix for UPT key name mismatch
            state_dict = {k.replace("upt_proj", "mvlpt_proj"):v for k, v in state_dict.items()}
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
