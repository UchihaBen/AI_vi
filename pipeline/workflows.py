"""ComfyUI workflows in API format, built in code so options (LoRAs, end frame,
lightning) can change the graph instead of patching a fixed JSON.

Node names/inputs target ComfyUI core + city96/ComfyUI-GGUF.
"""
from __future__ import annotations

SAVE_NODE = "save"


def sdxl_t2i(*, checkpoint: str, positive: str, negative: str, width: int, height: int,
             seed: int, steps: int, cfg: float, sampler: str, scheduler: str,
             loras: list[tuple[str, float]], batch: int, prefix: str) -> dict:
    wf: dict = {"ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}}}
    model, clip = ["ckpt", 0], ["ckpt", 1]
    for i, (name, strength) in enumerate(loras):
        nid = f"lora{i}"
        wf[nid] = {"class_type": "LoraLoader", "inputs": {
            "model": model, "clip": clip, "lora_name": name,
            "strength_model": strength, "strength_clip": strength}}
        model, clip = [nid, 0], [nid, 1]
    wf.update({
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": clip}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": clip}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": batch}},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": model, "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": scheduler,
            "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0], "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["ckpt", 2]}},
        SAVE_NODE: {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    })
    return wf


def dance_preprocess(*, drive_dir: str, d: dict, prefix: str) -> dict:
    """Driving dance frames -> pose skeleton frames + face crop frames.

    Needs: VideoHelperSuite, comfyui_controlnet_aux (DWPose), KJNodes, WanVideoWrapper.
    Output nodes: "save_pose", "save_face".
    """
    return {
        "drive": {"class_type": "VHS_LoadImagesPath", "inputs": {
            "directory": drive_dir, "image_load_cap": 0, "skip_first_images": 0, "select_every_nth": 1}},
        "dwpose": {"class_type": "DWPreprocessor", "inputs": {
            "image": ["drive", 0], "detect_hand": "enable" if d.get("detect_hand") else "disable",
            "detect_body": "enable", "detect_face": "disable", "resolution": int(d["pose_resolution"]),
            "bbox_detector": d["bbox_detector"], "pose_estimator": d["pose_estimator"],
            "scale_stick_for_xinsr_cn": "disable"}},
        "face_mask": {"class_type": "FaceMaskFromPoseKeypoints", "inputs": {"pose_kps": ["dwpose", 1], "person_index": 0}},
        "face_crop": {"class_type": "ImageCropByMaskAndResize", "inputs": {
            "image": ["drive", 0], "mask": ["face_mask", 0], "base_resolution": 512, "padding": 0,
            "min_crop_resolution": 128, "max_crop_resolution": 512}},
        "save_pose": {"class_type": "SaveImage", "inputs": {"images": ["dwpose", 0], "filename_prefix": f"{prefix}_pose"}},
        "save_face": {"class_type": "SaveImage", "inputs": {"images": ["face_crop", 0], "filename_prefix": f"{prefix}_face"}},
    }


def wan_animate_chunk(*, d: dict, positive: str, negative: str, width: int, height: int, length: int,
                      offset: int, seed: int, ref_image: str, pose_dir: str, face_dir: str,
                      continue_dir: str | None, continue_skip: int, prefix: str) -> dict:
    """One chunk of Wan 2.2 Animate. Chunks after the first get the last `overlap`
    frames of the previous chunk as continue_motion; those frames come back at the
    start of this chunk's output and must be dropped by the caller."""
    def load_dir(path: str, skip: int = 0) -> dict:
        return {"class_type": "VHS_LoadImagesPath", "inputs": {
            "directory": path, "image_load_cap": 0, "skip_first_images": skip, "select_every_nth": 1}}

    wf: dict = {
        "unet": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": d["unet"]}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet", 0], "lora_name": d["lora"], "strength_model": 1.0}},
        "shift": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["lora", 0], "shift": d["shift"]}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": d["text_encoder"], "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": d["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["clip", 0]}},
        "ref": {"class_type": "LoadImage", "inputs": {"image": ref_image}},
        "pose": load_dir(pose_dir),
        "face": load_dir(face_dir),
    }
    anim_inputs = {
        "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
        "width": width, "height": height, "length": length, "batch_size": 1,
        "continue_motion_max_frames": int(d["overlap"]), "video_frame_offset": offset,
        "reference_image": ["ref", 0], "pose_video": ["pose", 0], "face_video": ["face", 0],
    }
    if continue_dir:
        wf["prev"] = load_dir(continue_dir, continue_skip)
        anim_inputs["continue_motion"] = ["prev", 0]
    wf.update({
        "anim": {"class_type": "WanAnimateToVideo", "inputs": anim_inputs},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["shift", 0], "seed": seed, "steps": d["steps"], "cfg": d["cfg"],
            "sampler_name": d["sampler"], "scheduler": d["scheduler"],
            "positive": ["anim", 0], "negative": ["anim", 1], "latent_image": ["anim", 2], "denoise": 1.0}},
        "trim": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["sample", 0], "trim_amount": ["anim", 3]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["trim", 0], "vae": ["vae", 0]}},
        SAVE_NODE: {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    })
    return wf


def wan22_i2v(*, v: dict, positive: str, negative: str, width: int, height: int, length: int,
              seed: int, start_image: str, end_image: str | None, prefix: str) -> dict:
    """Wan 2.2 I2V A14B: high-noise expert for the first steps, low-noise for the rest.

    `v` is settings["video"]. With end_image the graph uses WanFirstLastFrameToVideo.
    Output: PNG frames (encoded to mp4 by ffmpeg afterwards).
    """
    preset = v["lightning_preset"] if v.get("lightning") else v["full_preset"]
    steps, cfg, shift, split = preset["steps"], preset["cfg"], preset["shift"], preset["split"]

    wf: dict = {
        "unet_h": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": v["unet_high"]}},
        "unet_l": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": v["unet_low"]}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": v["text_encoder"], "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": v["vae"]}},
    }
    model_h, model_l = ["unet_h", 0], ["unet_l", 0]
    if v.get("lightning"):
        wf["lora_h"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model_h, "lora_name": v["lora_high"], "strength_model": 1.0}}
        wf["lora_l"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model_l, "lora_name": v["lora_low"], "strength_model": 1.0}}
        model_h, model_l = ["lora_h", 0], ["lora_l", 0]
    wf.update({
        "shift_h": {"class_type": "ModelSamplingSD3", "inputs": {"model": model_h, "shift": shift}},
        "shift_l": {"class_type": "ModelSamplingSD3", "inputs": {"model": model_l, "shift": shift}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["clip", 0]}},
        "img_start": {"class_type": "LoadImage", "inputs": {"image": start_image}},
    })
    common = {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
              "width": width, "height": height, "length": length, "batch_size": 1,
              "start_image": ["img_start", 0]}
    if end_image:
        wf["img_end"] = {"class_type": "LoadImage", "inputs": {"image": end_image}}
        wf["cond"] = {"class_type": "WanFirstLastFrameToVideo", "inputs": {**common, "end_image": ["img_end", 0]}}
    else:
        wf["cond"] = {"class_type": "WanImageToVideo", "inputs": common}

    sampler = {"sampler_name": v["sampler"], "scheduler": v["scheduler"], "steps": steps, "cfg": cfg,
               "noise_seed": seed, "positive": ["cond", 0], "negative": ["cond", 1]}
    wf.update({
        "ks_high": {"class_type": "KSamplerAdvanced", "inputs": {
            **sampler, "model": ["shift_h", 0], "latent_image": ["cond", 2], "add_noise": "enable",
            "start_at_step": 0, "end_at_step": split, "return_with_leftover_noise": "enable"}},
        "ks_low": {"class_type": "KSamplerAdvanced", "inputs": {
            **sampler, "model": ["shift_l", 0], "latent_image": ["ks_high", 0], "add_noise": "disable",
            "start_at_step": split, "end_at_step": 10000, "return_with_leftover_noise": "disable"}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["ks_low", 0], "vae": ["vae", 0]}},
        SAVE_NODE: {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    })
    return wf
