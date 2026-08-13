from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn

LatentEmbSize = 768
# System 1 条件 latent 的内部维度。
MODEL_PATH_TO = Path(__file__).resolve().parents[4] / "checkpoints"
# checkpoint 根目录，用于加载 DepthAnythingV2 等额外权重。


def build_navdp(navdp_cfg, memory_size):
# 构建 NavDP 版本的 System 1；当前你的 DualVLN checkpoint 主要走 nextdit_async。
    from .navdp import NavDP_Policy_DPT_CriticSum_DAT

    navdp = NavDP_Policy_DPT_CriticSum_DAT(memory_size=memory_size, navdp_version=0.1)
    navdp.load_model()
    return navdp


def build_traj_dit(config):
# 构建 NextDiT 轨迹扩散模型，也就是你当前 checkpoint 的 System 1。
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    from .nextdit_crossattn_traj import NextDiTCrossAttn, NextDiTCrossAttnConfig

    dit = NextDiTCrossAttn(NextDiTCrossAttnConfig(latent_embedding_size=LatentEmbSize))
# Cross-Attention DiT：用 System 2 latent / memory tokens 作为条件生成轨迹。
    noise_scheduler = FlowMatchEulerDiscreteScheduler()
# 轨迹扩散采样使用的 scheduler。
    return dit, noise_scheduler


def build_depthanythingv2(config):
# 构建 DepthAnythingV2 的视觉 backbone，用于 System 1 提取 RGB 视觉特征。
    from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import (
        DepthAnythingV2,
    )

    model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
    DAv2_model = DepthAnythingV2(**model_configs['vits'])
# 使用 small/vits 版本的 DepthAnythingV2。
    DAv2_model.load_state_dict(
        torch.load(f'{MODEL_PATH_TO}/depth_anything_v2_metric_hypersim_vits.pth', map_location="cpu")
    )  # download from https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
# 加载本地 DepthAnythingV2 metric hypersim 权重。
    rgb_model = DAv2_model.pretrained
# 这里只取预训练视觉 backbone，用于提取图像特征，不直接预测深度。

    return rgb_model


class SinusoidalPositionalEncoding(nn.Module):
# System 1 轨迹 token 的时间步位置编码。
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
# 输入轨迹时间步编号，输出正余弦位置编码。
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


class MemoryEncoder(nn.Module):
# 异步双系统里的视觉记忆编码器：编码目标帧和当前帧的视觉 token。
    def __init__(self, hidden_size=384, num_heads=6, num_layers=3, max_len=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True, dropout=dropout
        )
# TransformerEncoderLayer 用于建模多个视觉 token 之间的关系。
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.memory_pos = nn.Parameter(torch.randn(max_len, hidden_size))
# 可学习位置编码，告诉模型 memory token 的相对位置。

    def forward(self, memory, memory_mask=None):
# 输入视觉 token memory，输出编码后的 memory 表示。
        """
        memory: (B, N, C)
        memory_mask: (B, N)
        """
        B, N, C = memory.shape
        pos = self.memory_pos[:N, :].unsqueeze(0).expand(B, -1, -1)  # (B, N, C)
        memory = memory + pos
        encoded_memory = self.encoder(memory, src_key_padding_mask=memory_mask)
        return encoded_memory


class QFormer(nn.Module):
# QFormer/rgb_resampler：用少量 query token 从大量视觉 token 中压缩出 memory tokens。
    def __init__(self, num_query=32, hidden_size=768, num_layers=3, num_heads=12):
        super().__init__()
        self.num_query = num_query
# 输出的压缩 query 数量。
        self.hidden_size = hidden_size

        self.query_tokens = nn.Parameter(torch.randn(num_query, hidden_size))
# 可学习 query token，用来主动读取视觉 memory。
        self.query_pos = nn.Parameter(torch.randn(num_query, hidden_size))
# query 的可学习位置编码。

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.visual_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, visual_feats, visual_attn_mask=None):
# 输入视觉特征，输出压缩后的 query 表示。
        B = visual_feats.size(0)

        query_tokens = self.query_tokens.unsqueeze(0).expand(B, -1, -1)
        query_tokens = query_tokens + self.query_pos.unsqueeze(0)

        out = self.decoder(query_tokens, visual_feats, memory_key_padding_mask=visual_attn_mask)
        return out


class InternVLAN1MetaModel:
# 给 Qwen2.5-VL backbone 挂载 InternVLA-N1 导航模块的 mixin。
    def __init__(self, config):
        super(InternVLAN1MetaModel, self).__init__(config)
        if hasattr(config, "system1"):
            self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))
# System 2 读取导航 latent 的可学习 query，数量由 config.n_query 决定。

            if 'nextdit' in config.system1:
# System 1 为 NextDiT 扩散轨迹模型。
                self.traj_dit, self.noise_scheduler = build_traj_dit(config)
# 轨迹 DiT 和扩散 scheduler。
                self.action_encoder = nn.Linear(3, 384, bias=True)
# 将 [dx, dy, dyaw] 编码成 DiT token 维度。
                self.pos_encoding = SinusoidalPositionalEncoding(384)
# 给 32 个未来轨迹步加入位置编码。
                self.action_decoder = nn.Linear(384, 3, bias=True)
# 将 DiT token 解码回 [dx, dy, dyaw]。
                self.cond_projector = nn.Sequential(
                    nn.Linear(3584, LatentEmbSize), nn.GELU(approximate="tanh"), nn.Linear(LatentEmbSize, LatentEmbSize)
                )
# 将 Qwen hidden size 3584 投影成 System 1 使用的 768 维条件 latent。

                if 'async' in config.system1:
# 异步双系统额外需要视觉 memory，用来结合旧 latent 和当前新观测。
                    self.rgb_model = build_depthanythingv2(config)
# RGB 视觉 backbone，用于提取当前帧/目标帧视觉特征。
                    self.memory_encoder = MemoryEncoder()
# 编码视觉 token 记忆。
                    self.rgb_resampler = QFormer()
# 将视觉 memory 压缩成少量条件 token。

            elif 'navdp' in config.system1:
# NavDP 版本 System 1。
                if 'async' in config.system1:
                    self.navdp = build_navdp(config, memory_size=2)
            else:
                raise NotImplementedError

    def initialize_vision_modules(self, model_args):
# 训练时初始化视觉/轨迹模块的入口，和 __init__ 中的结构保持一致。
        if 'nextdit' in model_args.system1:
# 初始化 NextDiT System 1。
            self.traj_dit, self.noise_scheduler = build_traj_dit(model_args)
            self.action_encoder = nn.Linear(3, 384, bias=True)
            self.pos_encoding = SinusoidalPositionalEncoding(384)
            self.action_decoder = nn.Linear(384, 3, bias=True)

            self.cond_projector = nn.Sequential(
                nn.Linear(3584, LatentEmbSize), nn.GELU(approximate="tanh"), nn.Linear(LatentEmbSize, LatentEmbSize)
            )

            if 'async' in model_args.system1:
                self.rgb_model = build_depthanythingv2(model_args)
                self.memory_encoder = MemoryEncoder()
                self.rgb_resampler = QFormer()
        elif 'navdp' in model_args.system1:
            if 'async' in model_args.system1:
                self.navdp = build_navdp(model_args, memory_size=2)
        else:
            raise NotImplementedError

        self.config.system1 = model_args.system1
        self.config.n_query = model_args.n_query
        if getattr(self, 'latent_queries', None) is None:
            print("random initiation the latent_queries !!!")
            self.latent_queries = nn.Parameter(torch.randn(1, self.config.n_query, self.config.hidden_size))


class InternVLAN1MetaForCausalLM(ABC):
# CausalLM 包装类使用的接口 mixin，用于访问内部 InternVLA-N1 模块。
    @abstractmethod
    def get_model(self):
        pass

    def get_mm_projector(self):
# 兼容多模态 projector 接口。
        return self.get_model().mm_projector

    def get_n_query(self):
# 返回 latent query 数量。
        return self.get_model().config.n_query

    def get_system1_type(self):
# 返回 System 1 类型，例如 nextdit_async。
        return self.get_model().config.system1

    def get_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
# 扩散训练/采样中根据 timestep 取 sigma 的辅助函数。
        sigmas = self.get_model().noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.get_model().noise_scheduler.timesteps.to(device=device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
