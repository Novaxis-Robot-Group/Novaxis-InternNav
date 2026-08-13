import copy
import itertools
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from collections import OrderedDict

from PIL import Image
from transformers import AutoProcessor

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import S2Output, split_and_clean, traj_to_actions

DEFAULT_IMAGE_TOKEN = "<image>"
# Qwen2.5-VL chat template 里的图像占位符，prompt 中每个 <image> 会对应一张输入图。


class InternVLAN1AsyncAgent:
# 实机部署使用的异步双系统 agent：低频 System 2 做高层决策，高频 System 1 做局部轨迹。
    def __init__(self, args):
        self.device = torch.device(args.device)
# 模型所在设备，例如 cuda:0。
        self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
# 每次运行保存 debug 图像和 LLM 输出的目录。
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"args.model_path{args.model_path}")
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": self.device},
        )
# 加载 InternVLA-N1 双系统模型；bfloat16 降低显存，flash_attention_2 加速注意力。
        self.model.eval()
# 切换到推理模式，关闭 dropout 等训练行为。
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(args.model_path)
# processor 负责把 chat 文本和图片打包成模型输入 tensor。
        self.processor.tokenizer.padding_side = 'left'
# 生成式模型常用 left padding，便于 batch generation 对齐。

        self.resize_w = args.resize_w
# System 2 输入图像 resize 宽度。
        self.resize_h = args.resize_h
# System 2 输入图像 resize 高度。
        self.num_history = args.num_history
# System 2 使用的历史图像数量。
        self.PLAN_STEP_GAP = args.plan_step_gap
# 异步规划间隔：每隔多少帧重新调用一次慢速 System 2。

        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Only output STOP when the instruction explicitly tells you to stop and the visible final target has been reached. For intermediate navigation instructions, do not output STOP; output a waypoint or a turn action instead."
# System 2 的核心 prompt：要求 VLM 根据指令和图像输出下一 waypoint 像素坐标或 STOP。
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
# 基础对话模板，每次 step_s2 会把 <instruction> 替换成真实导航指令。
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]

        self.actions2idx = OrderedDict(
# VLM 输出的文本动作符号到离散动作编号的映射。
            {
                'STOP': [0],
# 0 表示停止/不移动。
                "↑": [1],
# 1 表示前进一个小步长。
                "←": [2],
# 2 表示左转。
                "→": [3],
# 3 表示右转。
                "↓": [5],
# 5 表示低头/视角调整。
            }
        )

        self.rgb_list = []
# 历史 RGB 图像列表，供 System 2 使用历史观察。
        self.depth_list = []
# 历史 depth 列表，当前 realworld 主要在 System 1 局部轨迹里使用当前/目标深度。
        self.pose_list = []
# 历史 pose 列表，当前实机服务里 pose 主要是占位。
        self.episode_idx = 0
# 当前 episode 的帧编号。
        self.conversation_history = []
# 按 Qwen2.5-VL chat 格式组织的多轮对话历史。
        self.llm_output = ""
# 上一次 System 2 文本输出，可能是箭头动作或像素坐标。
        self.past_key_values = None
# 生成缓存，当前代码里保留但 generate 时没有启用 use_cache。
        self.last_s2_idx = -100
# 上一次调用 System 2 的帧编号，用于异步双系统间隔判断。
        self.last_instruction = None
# 上一次送入低层大脑的 instruction；变化时需要强制 System 2 重规划。

        # output
        self.output_action = None
# 缓存的离散动作输出。
        self.output_latent = None
# 缓存的 System 2 latent goal，供 System 1 生成轨迹。
        self.output_pixel = None
# 缓存的 System 2 像素目标点。
        self.pixel_goal_rgb = None
# 产生 pixel_goal 那一帧的 RGB，用于异步 System 1 对比当前帧。
        self.pixel_goal_depth = None
# 产生 pixel_goal 那一帧的 Depth，用于异步 System 1 对比当前帧。

    def reset(self):
# 开始新任务时重置 agent 内部状态，清空历史图像、对话和输出缓存。
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None
        self.last_instruction = None

        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None

        self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)

    def parse_actions(self, output):
# 把 VLM 输出文本中的 STOP/箭头符号解析成离散动作编号列表。
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def step_no_infer(self, rgb, depth, pose):
# 不调用 System 2 的中间帧处理：只保存当前 RGB，并推进 episode_idx。
        image = Image.fromarray(rgb).convert('RGB')
        image = image.resize((self.resize_w, self.resize_h))
        self.rgb_list.append(image)
        image.save(f"{self.save_dir}/debug_raw_{self.episode_idx:04d}.jpg")
        self.episode_idx += 1

    def trajectory_tovw(self, trajectory, kp=1.0):
# 将轨迹末端粗略转换成线速度和角速度的辅助函数，当前主链路没有直接使用。
        subgoal = trajectory[-1]
        linear_vel, angular_vel = kp * np.linalg.norm(subgoal[:2]), kp * subgoal[2]
        linear_vel = np.clip(linear_vel, 0, 0.5)
        angular_vel = np.clip(angular_vel, -0.5, 0.5)
        return linear_vel, angular_vel

    def step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
# agent 单步入口：server 每收到一帧 RGB-D，就调用这里得到动作或轨迹。
        dual_sys_output = S2Output()
# 标准输出容器，可能包含 output_action、output_trajectory 或 output_pixel。
        instruction = str(instruction or "").strip()
        instruction_changed = self.last_instruction is not None and instruction != self.last_instruction
        if instruction_changed:
            self.output_action = None
            self.output_latent = None
            self.output_pixel = None
            self.pixel_goal_rgb = None
            self.pixel_goal_depth = None
            self.last_s2_idx = -100
# 新子任务到来时丢弃旧动作/旧 latent，避免继续执行上一条 instruction 的缓存轨迹。
        no_output_flag = self.output_action is None and self.output_latent is None
# 如果既没有动作也没有 latent，说明必须调用 System 2 重新规划。
        if (self.episode_idx - self.last_s2_idx > self.PLAN_STEP_GAP) or look_down or no_output_flag:
# 满足间隔、低头视角或无缓存输出时，调用慢速 System 2。
            self.output_action, self.output_latent, self.output_pixel = self.step_s2(
                rgb, depth, pose, instruction, intrinsic, look_down
            )
            if not look_down:
                self.last_instruction = instruction
            self.last_s2_idx = self.episode_idx
# 记录最近一次 System 2 推理对应的帧编号。
            dual_sys_output.output_pixel = self.output_pixel
            self.pixel_goal_rgb = copy.deepcopy(rgb)
# 保存 System 2 决策发生时的 RGB，供异步 System 1 作为目标帧上下文。
            self.pixel_goal_depth = copy.deepcopy(depth)
# 保存 System 2 决策发生时的 Depth。
        else:
            self.step_no_infer(rgb, depth, pose)
# 没到 System 2 间隔时，只记录当前帧，随后用旧 latent 调 System 1。

        if self.output_action is not None:
# 如果 System 2 直接输出离散动作，就直接返回给 server/client。
            dual_sys_output.output_action = copy.deepcopy(self.output_action)
            self.output_action = None
# 动作返回一次后清空，避免下一帧重复执行同一批动作。
        elif self.output_latent is not None:
# 如果 System 2 输出的是像素目标/latent，则调用 System 1 生成连续轨迹。
            processed_pixel_rgb = np.array(Image.fromarray(self.pixel_goal_rgb).resize((224, 224))) / 255
# System 2 决策帧 RGB，resize 到 System 1 使用的 224x224 并归一化到 0-1。
            processed_pixel_depth = np.array(Image.fromarray(self.pixel_goal_depth).resize((224, 224)))
# System 2 决策帧 depth，resize 到 224x224。
            processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255
# 当前帧 RGB，作为异步 System 1 的最新观测。
            processed_depth = np.array(Image.fromarray(depth).resize((224, 224)))
# 当前帧 depth。
            rgbs = (
                torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)])
                .unsqueeze(0)
                .to(self.device)
            )
            depths = (
                torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)])
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self.device)
            )
            trajectories = self.step_s1(self.output_latent, rgbs, depths)
# System 1 根据 latent goal 和当前/目标 RGB-D 生成候选轨迹。

            dual_sys_output.output_trajectory = traj_to_actions(trajectories, use_discrate_action=False)
# 将多条候选增量轨迹还原并平均，返回连续局部 trajectory。

        return dual_sys_output

    def step_s2(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
# System 2 慢速推理：使用 VLM 根据图像、历史和指令输出动作或像素目标。
        image = Image.fromarray(rgb).convert('RGB')
        if not look_down:
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
# 保存当前 RGB 到历史列表，后续 System 2 可引用历史观察。
            image.save(f"{self.save_dir}/debug_raw_{self.episode_idx:04d}.jpg")
# 保存 debug 输入图，方便复现某一帧的模型输出。
        else:
            image.save(f"{self.save_dir}/debug_raw_{self.episode_idx:04d}_look_down.jpg")
        if not look_down:
            self.conversation_history = []
# 每次非 look_down 的 System 2 推理重新组织对话历史。
            self.past_key_values = None

            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
# 把 prompt 中的 <instruction> 替换成当前任务语言指令。
            cur_images = self.rgb_list[-1:]
# 当前帧图像一定会作为最后一张输入图。
            if self.episode_idx == 0:
                history_id = []
            else:
                history_id = np.unique(np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)).tolist()
# 从历史帧中均匀采样 num_history 张图，避免 prompt 过长。
                placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'
# 在 prompt 中插入历史观察图像占位符。

            history_id = sorted(history_id)
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
# System 2 实际输入图像：历史帧 + 当前帧。
            input_img_id = 0
            self.episode_idx += 1
# System 2 消耗当前帧后，episode 帧编号前进。
        else:
            self.input_images.append(image)
# look_down 分支会把低头图像追加到本轮输入图中。
            input_img_id = -1
            assert self.llm_output != "", "Last llm_output should not be empty when look down"
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
            )

        prompt = self.conjunctions[0] + DEFAULT_IMAGE_TOKEN
# 给当前观察加一个自然语言引导，例如 "you can see <image>"。
        sources[0]["value"] += f" {prompt}."
        prompt_instruction = copy.deepcopy(sources[0]["value"])
        parts = split_and_clean(prompt_instruction)
# 将 prompt 按 <image> 拆分成文本片段和图像占位符。

        content = []
        for i in range(len(parts)):
            if parts[i] == "<image>":
                content.append({"type": "image", "image": self.input_images[input_img_id]})
# 遇到 <image> 时，放入对应 PIL 图像。
                input_img_id += 1
            else:
                content.append({"type": "text", "text": parts[i]})
# 普通文本片段直接作为 chat content。

        self.conversation_history.append({'role': 'user', 'content': content})

        text = self.processor.apply_chat_template(self.conversation_history, tokenize=False, add_generation_prompt=True)
# 使用 Qwen2.5-VL chat template 把多模态对话转成模型可读文本。

        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt").to(self.device)
# processor 同时编码文本和图像，得到 input_ids、pixel_values、image_grid_thw 等。
        t0 = time.time()
        with torch.no_grad():
# System 2 是纯推理，不需要梯度。
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                # use_cache=True,
                # past_key_values=self.past_key_values,
                return_dict_in_generate=True,
                # raw_input_ids=copy.deepcopy(inputs.input_ids),
            )
        output_ids = outputs.sequences

        t1 = time.time()
        self.llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
# 解码新生成的 token，得到 VLM 输出文本。
        with open(f"{self.save_dir}/llm_output_{self.episode_idx:04d}.txt", 'w') as f:
            f.write(self.llm_output)
        self.last_output_ids = copy.deepcopy(output_ids[0])
        self.past_key_values = copy.deepcopy(outputs.past_key_values)
        print(f"output {self.episode_idx}  {self.llm_output} cost: {t1 - t0}s")
        if bool(re.search(r'\d', self.llm_output)):
# 如果输出里包含数字，认为 System 2 输出的是像素目标坐标。
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            pixel_goal = [int(coord[1]), int(coord[0])]
# 将模型输出坐标整理成 [x, y] 或代码约定的像素目标格式。
            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
            pixel_values = inputs.pixel_values
            t0 = time.time()
            with torch.no_grad():
                traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)
# 从 VLM hidden states 中提取 System 1 所需的 latent goal。
                return None, traj_latents, pixel_goal

        else:
# 如果输出里没有数字，则按 STOP/箭头离散动作解析。
            action_seq = self.parse_actions(self.llm_output)
            return action_seq, None, None

    def step_s1(self, latent, rgb, depth):
# System 1 快速轨迹生成：latent goal + 当前/目标 RGB-D -> 多条候选轨迹。
        all_trajs = self.model.generate_traj(latent, rgb, depth)
        return all_trajs

    def get_debug_snapshot(self):
        # 导出当前 agent 的文本调试信息，供 server 保存到实验日志和 Web viewer 展示。
        def clean_content_item(item):
            if item.get("type") == "text":
                return {"type": "text", "text": item.get("text", "")}
            if item.get("type") == "image":
                image = item.get("image")
                size = getattr(image, "size", None)
                return {"type": "image", "image": f"<image size={size}>"}
            return {str(k): str(v) for k, v in item.items()}

        cleaned_history = []
        for message in self.conversation_history:
            content = message.get("content", [])
            if isinstance(content, list):
                cleaned_content = [clean_content_item(item) for item in content]
            else:
                cleaned_content = str(content)
            cleaned_history.append({"role": message.get("role", ""), "content": cleaned_content})

        return {
            "episode_idx": self.episode_idx,
            "last_s2_idx": self.last_s2_idx,
            "last_instruction": self.last_instruction,
            "llm_output": self.llm_output,
            "conversation_history": cleaned_history,
            "num_rgb_history": len(self.rgb_list),
            "num_input_images": len(getattr(self, "input_images", [])),
            "save_dir": self.save_dir,
        }
