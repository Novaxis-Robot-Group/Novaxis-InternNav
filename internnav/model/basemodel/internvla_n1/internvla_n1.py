from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .internvla_n1_arch import InternVLAN1MetaForCausalLM, InternVLAN1MetaModel

TRAJ_TOKEN_INDEX = 151667
# 轨迹 latent query 使用的特殊 token id，用来在 VLM 序列里插入可学习导航查询。
IMAGE_TOKEN_INDEX = 151655
# Qwen2.5-VL 图像 token id，后续会被真实图像视觉 embedding 替换。
_RESNET_MEAN = [0.485, 0.456, 0.406]
# System 1 视觉 backbone 的 RGB 归一化均值。
_RESNET_STD = [0.229, 0.224, 0.225]
# System 1 视觉 backbone 的 RGB 归一化方差。


class InternVLAN1ModelConfig(Qwen2_5_VLConfig):
# InternVLA-N1 配置类：在 Qwen2.5-VL 配置基础上扩展导航相关字段。
    model_type = "internvla_n1"
# transformers 根据 model_type 找到并加载 InternVLA-N1 自定义模型。

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)


class InternVLAN1Model(InternVLAN1MetaModel, Qwen2_5_VLModel):
# InternVLA-N1 backbone：Qwen2.5-VL 主体 + InternVLA-N1 导航模块。
    config_class = InternVLAN1ModelConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(InternVLAN1Model, self).__init__(config)


class InternVLAN1ForCausalLM(Qwen2_5_VLForConditionalGeneration, InternVLAN1MetaForCausalLM):
# 实际 from_pretrained 加载的模型类：既能生成文本，也能生成导航 latent 和轨迹。
    config_class = InternVLAN1ModelConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type == "internvla_n1"

        self.model = InternVLAN1Model(config)
# 内部模型主体，包含 Qwen2.5-VL 和 System 1 轨迹模块。
        self.rope_deltas = None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
# 语言模型 head，用于 System 2 输出动作符号或像素坐标文本。
        # Initialize weights and apply final processing
        self.post_init()

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)
# 注册 RGB 归一化均值/方差 buffer，System 1 提取视觉特征时使用。

    def get_model(self):
# 返回内部 backbone，供 meta mixin 和外部 agent 访问导航模块。
        return self.model

    def forward(
# 训练/评估前向：支持 Qwen2.5-VL 文本生成，也支持训练 System 1 轨迹分支。
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        t_s_pos: Optional[list] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        traj_images: Optional[torch.Tensor] = None,
        traj_depths: Optional[torch.Tensor] = None,
        video_frame_num: Optional[torch.Tensor] = None,
        traj_poses: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
# 文本 token id 转为 embedding。
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
# Qwen2.5-VL 视觉塔把输入图像编码成视觉 token embedding。
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
# 找到文本序列中所有 image token 位置。
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
# 用真实图像 embedding 替换掉 <image> token embedding。

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            n_traj_tokens = (input_ids == TRAJ_TOKEN_INDEX).sum().item()
# 统计序列里轨迹 query token 数量。
            traj_idx = input_ids == TRAJ_TOKEN_INDEX
# 找到轨迹 query token 位置。
            latent_queries = self.get_model().latent_queries.repeat(input_ids.shape[0], 1, 1)
# 可学习 latent query，负责从 VLM 上下文中读取导航目标表示。
            H = latent_queries.shape[-1]
            latent_queries = latent_queries.contiguous().view(-1, H)
            if n_traj_tokens != 0:
                inputs_embeds[traj_idx] = latent_queries
# 训练时将特殊轨迹 token 替换成可学习 latent query embedding。

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
# 调用 Qwen2.5-VL backbone，得到 hidden states。
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
# logits 用于文本生成或语言建模损失。

        loss = None
        if labels is not None:
# labels 存在时说明处于训练/评估损失计算流程。
            traj_hidden_states = []
# 从 VLM hidden states 中抽取轨迹 query 对应的隐藏状态。
            for b in range(hidden_states.shape[0]):
                traj_hidden_states.append(hidden_states[b, t_s_pos[b] : t_s_pos[b] + self.config.n_query, :])

            traj_hidden_states = torch.stack(traj_hidden_states, dim=0)
            traj_hidden_states = traj_hidden_states.unsqueeze(1).repeat(1, traj_poses.size(1), 1, 1).flatten(0, 1)
            loss_mask = torch.arange(traj_images.size(1), device=self.device).expand(
                traj_images.size(0), traj_images.size(1)
            ) < video_frame_num.unsqueeze(1)

            if 'nextdit' in self.get_system1_type():
                if 'async' in self.get_system1_type():
                    cur_images = traj_images.flatten(0, 1)
                    pix_goal_images = traj_images[:, 0:1].repeat(1, traj_images.size(1), 1, 1, 1).flatten(0, 1)
                    bsz = cur_images.size(0)
                    images_dp = torch.stack([pix_goal_images, cur_images], dim=1).permute(0, 1, 4, 2, 3)
                    images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std

                    images_dp_feat = (
                        self.get_model()
                        .rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1))[0]
                        .unflatten(dim=0, sizes=(bsz, -1))
                    )

                    memory_feat = self.get_model().memory_encoder(
                        images_dp_feat.flatten(1, 2)
                    )  # [bs*select_size,512,384]
                    memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
                    memory_tokens = self.get_model().rgb_resampler(memory_feat)

                    traj_hidden_states = self.get_model().cond_projector(traj_hidden_states)
                    latents = torch.cat([memory_tokens, traj_hidden_states], dim=1)
                else:
                    traj_hidden_states = self.get_model().cond_projector(traj_hidden_states)
                    latents = traj_hidden_states

                relative_poses = traj_poses.flatten(0, 1)
                bsz = relative_poses.shape[0]
                noise = torch.randn(relative_poses.shape, device=relative_poses.device, dtype=relative_poses.dtype)
                u = torch.rand(size=(bsz,), device="cpu")
                indices = (u * self.get_model().noise_scheduler.config.num_train_timesteps).long()
                timesteps = self.get_model().noise_scheduler.timesteps[indices].to(device=latents.device)
                sigmas = self.get_sigmas(
                    timesteps, latents.device, n_dim=relative_poses.shape[-1], dtype=relative_poses.dtype
                )

                noisy_trajectory = (1 - sigmas) * relative_poses + sigmas * noise
                action_features = self.get_model().action_encoder(noisy_trajectory)
                pos_ids = torch.arange(relative_poses.shape[1]).reshape(1, -1).repeat(bsz, 1).to(relative_poses.device)
                pos_embed = self.get_model().pos_encoding(pos_ids)
                action_features += pos_embed

                noise_pred = self.get_model().traj_dit(
                    x=action_features,
                    timestep=timesteps,
                    z_latents=latents,
                )
                noise_pred = self.get_model().action_decoder(noise_pred)
                target = noise - relative_poses
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                mask = loss_mask.flatten(0, 1)[:, None, None]
                masked_loss = loss * mask
                loss = masked_loss.sum() / mask.sum() / (loss.shape[1] * loss.shape[2])
            elif 'navdp' in self.get_system1_type():
                if 'async' in self.get_system1_type():
                    cur_images = traj_images.flatten(0, 1)
                    cur_depths = traj_depths.flatten(0, 1)
                    pix_goal_images = traj_images[:, 0:1].repeat(1, traj_images.size(1), 1, 1, 1).flatten(0, 1)
                    pix_goal_depths = traj_depths[:, 0:1].repeat(1, traj_depths.size(1), 1, 1).flatten(0, 1)
                    images_dp = torch.stack([pix_goal_images, cur_images], dim=1)  # (bs*select_size, 2, 224, 224, 3)
                    depths_dp = torch.stack([pix_goal_depths, cur_depths], dim=1).unsqueeze(
                        -1
                    )  # (bs*select_size, 2, 224, 224, 1)
                    pred_pg, noise = self.model.navdp.forward_vlm_traj(
                        traj_hidden_states, images_dp, depths_dp, tensor_label_actions=traj_poses
                    )
                    pg_action_loss = (pred_pg - noise).square()
                    mask = loss_mask.flatten(0, 1)[:, None, None]
                    masked_loss = pg_action_loss * mask
                    loss = masked_loss.sum() / mask.sum() / (pg_action_loss.shape[1] * pg_action_loss.shape[2])

            else:
                raise NotImplementedError

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate_latents(self, input_ids, pixel_values, image_grid_thw):
# System 2 到 System 1 的桥：从 VLM 上下文中抽取 latent goal。
        input_ids.to(self.get_model().device)
        with torch.no_grad():
            text_embeds = self.get_model().embed_tokens(input_ids)
# 将 System 2 已生成的 token 序列重新编码成 embedding。
        latent_queries = self.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)
# 可学习导航查询 token，数量由 config.n_query 决定，你当前 checkpoint 是 4。
        image_idx = input_ids == IMAGE_TOKEN_INDEX
# 找到图像 token 位置，后续用真实视觉 embedding 替换。
        N_QUERY = self.get_n_query()
        input_ids = torch.cat([input_ids, torch.tensor([[TRAJ_TOKEN_INDEX] * N_QUERY]).to(input_ids.device)], dim=1)
# 在序列末尾拼接 N_QUERY 个轨迹 token，用它们读取导航 latent。

        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).unsqueeze(0)
# 重新提取输入图像的 Qwen-VL 视觉特征。

        text_embeds[image_idx] = image_embeds.to(text_embeds.device)[: image_idx.sum(), :]
# 将 <image> 位置替换成真实图像视觉 embedding。

        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)
# 得到 [X; Z]：X 是语言/历史图像/当前图像上下文，Z 是导航查询 token。

        position_ids, _ = self.get_rope_index(input_ids, image_grid_thw)
# 计算多模态 RoPE 位置编码。
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=text_embeds,
                position_ids=position_ids,
                # attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states[-1][:, -N_QUERY:, :]
# 取最后 N_QUERY 个 hidden states，作为 Z' latent goal 交给 System 1。

        return hidden_states

    def generate_traj(
# System 1 推理：用 latent goal 和 RGB-D 观测生成候选连续轨迹。
        self,
        traj_latents,
        images_dp,
        depths_dp=None,
        predict_step_nums=32,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 10,
# 扩散采样去噪步数，默认 10。
        num_sample_trajs: int = 32,
# 每次采样的候选轨迹数量，默认 32 条。
    ):
        if 'nextdit' in self.get_system1_type():
# 你当前 checkpoint 的 System 1 类型是 nextdit_async，会走这个分支。
            scheduler = FlowMatchEulerDiscreteScheduler()
# Flow Matching Euler scheduler，用于轨迹扩散去噪。
            device = traj_latents.device
            dtype = traj_latents.dtype

            traj_latents = self.get_model().cond_projector(traj_latents)
# 将 Qwen hidden size 的 latent goal 投影到 NextDiT 使用的条件维度。
            if 'async' in self.get_system1_type():
# async 表示 System 2 低频更新，System 1 高频使用旧 latent + 当前新观测。
                with torch.no_grad():
                    images_dp = images_dp.permute(0, 1, 4, 2, 3)
# RGB 从 [B, T, H, W, C] 转成视觉 backbone 需要的 [B, T, C, H, W]。
                    images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std
# 按 ImageNet/ResNet 风格均值方差归一化。
                    self.get_model().rgb_model.to(dtype)
                    images_dp_feat = (
                        self.get_model()
                        .rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1).to(dtype))[0]
                        .unflatten(dim=0, sizes=(1, -1))
                    )
                    memory_feat = self.get_model().memory_encoder(
                        images_dp_feat.flatten(1, 2)
                    )  # [bs*select_size,512,384]
# memory_encoder 编码目标帧和当前帧的视觉 token，用于处理异步情况下环境变化。
                    memory_feat = torch.cat([images_dp_feat.flatten(1, 2), memory_feat], dim=-1)
                    memory_tokens = self.get_model().rgb_resampler(memory_feat)
# QFormer/rgb_resampler 将大量视觉 token 压缩成紧凑 memory tokens。
                hidden_states = torch.cat([memory_tokens, traj_latents], dim=1)
# System 1 的条件 = 视觉 memory + System 2 latent goal。
            else:
                hidden_states = traj_latents
            hidden_states_null = torch.zeros_like(hidden_states, device=device, dtype=dtype)
# classifier-free guidance 的无条件分支。
            hidden_states_input = torch.cat([hidden_states_null, hidden_states], 0)
# 将无条件和有条件分支拼在一起，一次前向里共同计算。
            batch_size = traj_latents.shape[0]
            latent_size = predict_step_nums
# 轨迹预测步数，默认 32。
            latent_channels = 3
# 每个轨迹点包含 [dx, dy, dyaw] 三个分量。

            latents = randn_tensor(
                shape=(batch_size * num_sample_trajs, latent_size, latent_channels),
                generator=None,
                device=device,
                dtype=dtype,
            )
# 从随机噪声初始化候选轨迹。

            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
# 设置扩散采样时间步。

            hidden_states_input = hidden_states_input.repeat_interleave(num_sample_trajs, dim=0)
# 每个条件重复 num_sample_trajs 次，对应采样多条候选轨迹。

            for t in scheduler.timesteps:
# 逐步去噪：从随机轨迹变成可通行局部轨迹。
                latent_features = self.get_model().action_encoder(latents)
# 将当前 noisy trajectory 的 [dx, dy, dyaw] 编码成 DiT token。
                pos_ids = (
                    torch.arange(latent_features.shape[1])
                    .reshape(1, -1)
                    .repeat(batch_size, 1)
                    .to(latent_features.device)
                )
                pos_embed = self.get_model().pos_encoding(pos_ids)
                latent_features += pos_embed  # [num_sample_trajs, t, 384]
# 给轨迹时间步加入位置编码，让 DiT 知道第几个未来点。
                latent_model_input = latent_features.repeat(2, 1, 1)
# 对应 classifier-free guidance 的无条件/有条件两份输入。
                if hasattr(scheduler, "scale_model_input"):
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # predict noise model_output
                noise_pred = self.get_model().traj_dit(
# NextDiT 根据 noisy trajectory 和条件 hidden states 预测噪声。
                    x=latent_model_input,
                    timestep=t.unsqueeze(0)
                    .expand(latent_model_input.shape[0])
                    .to(latent_model_input.device, torch.long),
                    z_latents=hidden_states_input,
                )

                noise_pred = self.get_model().action_decoder(noise_pred)
# 将 DiT 输出解码回 [dx, dy, dyaw] 噪声预测。

                # perform guidance
                noise_pred_uncond, noise_pred = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
# classifier-free guidance：用条件分支增强任务相关轨迹。

                # compute previous: x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample
# 执行一步去噪更新。
            return latents
# 返回多条候选增量轨迹，后续 vln_utils.traj_to_actions 会还原和平均。

        elif 'navdp' in self.get_system1_type():
# NavDP 版本 System 1 分支；你当前 DualVLN checkpoint 不走这里。
            if 'async' in self.get_system1_type():
                all_trajs = self.model.navdp.predict_pointgoal_action_async(
                    traj_latents.to(self.get_model().device), images_dp, depths_dp
                )
            else:
                all_trajs = self.model.navdp.predict_pointgoal_action(traj_latents.to(self.get_model().device))
            return all_trajs
