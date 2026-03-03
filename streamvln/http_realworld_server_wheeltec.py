"""
http_realworld_server_wheeltec.py
==================================
StreamVLN inference server for Wheeltec robot deployment.

Forked from http_realworld_server.py with the following changes:
  1. Camera intrinsics: Orbbec Gemini 336L (or Astra S) instead of sim values
  2. rgb_height: 0.35 m  (wheeled robot, not quadruped)
  3. Instruction: accepted from each POST request body (not hardcoded)
  4. Camera: selectable via --camera CLI argument

Run on the GPU workstation:
    python streamvln/http_realworld_server_wheeltec.py \
        --model_path checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world \
        --camera gemini_336l \
        --num_history 8 \
        --num_future_steps 4 \
        --num_frames 32
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import transformers
from datetime import datetime
from flask import Flask, jsonify, request
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamvln.streamvln_agent import VLNEvaluator
from model.stream_video_vln import StreamVLNForCausalLM

# ── Camera intrinsic configs ──────────────────────────────────────────────────
# Gemini 336L: native 1280×800, RGB FOV H94°×V68°, Depth FOV H90°×V65°
# Optimal depth range: 0.25-6 m  |  Max: 0.17-20 m
CAMERA_CONFIGS = {
    "gemini_336l": {
        # Native 1280×800, intrinsics computed from FOV (calibrate for best accuracy)
        "width": 1280,
        "height": 800,
        "fx": 597.0,   # ≈ 640 / tan(47°)
        "fy": 593.0,   # ≈ 400 / tan(34°)
        "cx": 640.0,
        "cy": 400.0,
        "rgb_height": 0.35,
    },
    "gemini_336l_720": {
        # SDK configured to 1280×720 output; calibrated values from InternNav project
        "width": 1280,
        "height": 720,
        "fx": 607.45,
        "fy": 607.40,
        "cx": 639.19,
        "cy": 361.75,
        "rgb_height": 0.35,
    },
    "astra_s": {
        "width": 640,
        "height": 480,
        "fx": 570.3,
        "fy": 570.3,
        "cx": 319.5,
        "cy": 239.5,
        "rgb_height": 0.35,
    },
}

DEFAULT_INSTRUCTION = "Navigate to the destination."

# ── Flask app and global state ────────────────────────────────────────────────
app = Flask(__name__)

action_seq         = np.zeros(4)
idx                = 0
terminate          = False
total_generate_time = 0.0
start_time         = time.time()
output_dir         = ""
llm_output         = ""


def build_intrinsic_matrix(cfg: dict) -> np.ndarray:
    fx, fy, cx, cy = cfg["fx"], cfg["fy"], cfg["cx"], cfg["cy"]
    return np.array([
        [fx,   0.,  cx,  0.],
        [ 0.,  fy,  cy,  0.],
        [ 0.,   0.,  1.,  0.],
        [ 0.,   0.,  0.,  1.],
    ], dtype=np.float64)


def annotate_image(frame_idx, image_arr, t_start, gen_time, action_str, out_dir):
    img = Image.fromarray(image_arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    lines = [
        f"Frame      : {frame_idx}",
        f"Elapsed    : {time.time() - t_start:.2f}s",
        f"Infer time : {gen_time:.2f}s",
        f"Actions    : {action_str}",
    ]
    pad = 10
    line_h = 26
    box_w = max(draw.textlength(l, font=font) for l in lines) + 2 * pad
    box_h = len(lines) * line_h + 2 * pad
    draw.rectangle([10, 10, 10 + box_w, 10 + box_h], fill="black")
    y = 10 + pad
    for line in lines:
        draw.text((10 + pad, y), line, fill="white", font=font)
        y += line_h

    img.save(os.path.join(out_dir, f"rgb_{frame_idx}_annotated.png"))


# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.route("/eval_vln", methods=["POST"])
def eval_vln():
    global action_seq, idx, terminate, total_generate_time
    global output_dir, start_time, llm_output

    # ── Parse request ─────────────────────────────────────────────────────────
    image_file = request.files["image"]
    data       = json.loads(request.form["json"])
    policy_init  = data.get("reset", False)
    instruction  = data.get("instruction", DEFAULT_INSTRUCTION)

    image_pil = Image.open(image_file.stream).convert("RGB")
    image_bgr = np.asarray(image_pil)[..., ::-1]  # RGB → BGR for evaluator

    # ── Reset on new episode ──────────────────────────────────────────────────
    if policy_init:
        start_time          = time.time()
        total_generate_time = 0.0
        terminate           = False
        idx                 = 0
        llm_output          = ""
        output_dir = "runs_wheeltec" + datetime.now().strftime("%m-%d-%H%M")
        os.makedirs(output_dir, exist_ok=True)
        evaluator.reset_memory()
        print(f"[server] episode reset – instruction: {instruction!r}")
        print(f"[server] saving frames to {output_dir}/")

    idx += 1

    if terminate:
        print("[server] task already finished – returning STOP")
        return jsonify({"action": [0]})

    # ── Run evaluator for num_future_steps steps ──────────────────────────────
    for _ in range(4):
        t1 = time.time()
        return_action, generate_time, return_llm_out = evaluator.step(
            0,
            image_bgr,
            instruction,
            run_model=(evaluator.step_id % 4 == 0),
        )
        if return_llm_out is not None:
            llm_output = return_llm_out
        if generate_time > 0:
            total_generate_time = generate_time
        action_seq = action_seq if return_action is None else return_action
        if 0 in action_seq:
            terminate = True
        evaluator.step_id += 1
        print(f"[server] step {evaluator.step_id} cost {time.time() - t1:.3f}s")

    # ── Format for logging ────────────────────────────────────────────────────
    sym_map = {"1": "↑", "2": "←", "3": "→", "0": "STOP"}
    str_action = "".join(sym_map.get(c, c) for c in
                         "".join(str(int(a)) for a in action_seq))
    if idx > 1 and total_generate_time > 0.5:
        total_generate_time -= 0.3

    annotate_image(idx, image_bgr[:, :, ::-1],  # BGR → RGB for PIL
                   start_time, total_generate_time, str_action, output_dir)

    if len(action_seq) == 0:
        print("[server] empty action sequence – returning STOP")
        return jsonify({"action": [0]})

    return jsonify({"action": action_seq.tolist()})


# ── Entry point ───────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="StreamVLN HTTP inference server for Wheeltec deployment"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world",
        help="Path to the StreamVLN model checkpoint",
    )
    parser.add_argument(
        "--camera",
        default="gemini_336l",
        choices=list(CAMERA_CONFIGS.keys()),
        help="Camera model: gemini_336l (default) or astra_s",
    )
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames",       type=int, default=32)
    parser.add_argument("--num_history",      type=int, default=8)
    parser.add_argument(
        "--model_max_length", type=int, default=4096,
        help="Maximum token sequence length",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port",   type=int, default=8909)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cam_cfg = CAMERA_CONFIGS[args.camera]

    print(f"[server] loading model from {args.model_path!r}")
    print(f"[server] camera: {args.camera} | "
          f"intrinsics: fx={cam_cfg['fx']}, fy={cam_cfg['fy']}, "
          f"cx={cam_cfg['cx']}, cy={cam_cfg['cy']}")

    # ── Load model ────────────────────────────────────────────────────────────
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.model_max_length,
        padding_side="right",
    )
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model  = StreamVLNForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=False,
    )
    model.model.num_history = args.num_history
    model.reset(1)
    model.requires_grad_(False)
    model.to(args.device)
    model.eval()

    # ── Build sensor config with Wheeltec camera intrinsics ──────────────────
    vln_sensor_config = {
        "rgb_height":       cam_cfg["rgb_height"],
        "camera_intrinsic": build_intrinsic_matrix(cam_cfg),
    }

    evaluator = VLNEvaluator(
        vln_sensor_config,
        model=model,
        tokenizer=tokenizer,
        args=args,
    )

    # Warm-up forward pass
    dummy = np.zeros((cam_cfg["height"], cam_cfg["width"], 3), dtype=np.uint8)
    evaluator.step(0, dummy, DEFAULT_INSTRUCTION, run_model=True)
    print(f"[server] warm-up done – listening on 0.0.0.0:{args.port}")

    app.run(host="0.0.0.0", port=args.port)
