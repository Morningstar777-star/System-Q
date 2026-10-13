"""
Deploy System Q Interactive Demo to Hugging Face Spaces.
Repo: LuciferMorningstar777/system-q-demo
"""

import os
import shutil
import argparse
from huggingface_hub import HfApi, create_repo

SPACE_README = """---
title: System Q Demo
emoji: ⚡
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Zero-Token Single-Pass System 1 Decision Architecture from Qwen2.5-3B
---

# ⚡ System Q: Fast System 1 Decision Engine
Interactive demo for System Q (Qwen2.5-3B Direct Scoring + Canonical Anchoring).
GitHub: [Morningstar777-star/System-Q](https://github.com/Morningstar777-star/System-Q)
Model: [LuciferMorningstar777/system-q-3b-decision](https://huggingface.co/LuciferMorningstar777/system-q-3b-decision)
"""

SPACE_REQUIREMENTS = """gradio>=4.44.0
torch
transformers>=4.45.0
peft>=0.13.0
bitsandbytes>=0.43.0
accelerate>=0.34.0
sentencepiece
"""


def deploy(token: str, space_id: str = "LuciferMorningstar777/system-q-demo"):
    api = HfApi(token=token)
    
    print(f"Ensuring Space repository exists: {space_id}...")
    create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        token=token
    )
    
    deploy_dir = "deploy_hf_space"
    os.makedirs(deploy_dir, exist_ok=True)
    
    # 1. Write Space README.md
    with open(os.path.join(deploy_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(SPACE_README)
        
    # 2. Write requirements.txt
    with open(os.path.join(deploy_dir, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(SPACE_REQUIREMENTS)
        
    # 3. Copy app.py
    shutil.copy("app.py", os.path.join(deploy_dir, "app.py"))
    
    # 4. Copy src/
    src_dest = os.path.join(deploy_dir, "src")
    if os.path.exists(src_dest):
        shutil.rmtree(src_dest)
    shutil.copytree("src", src_dest)
    
    print(f"Uploading Space files from {deploy_dir} to {space_id}...")
    api.upload_folder(
        folder_path=deploy_dir,
        repo_id=space_id,
        repo_type="space",
        commit_message="Deploy System Q Gradio Space Demo"
    )
    
    print(f"\n[SUCCESS] Space deployed successfully! Live at: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HF Write Token")
    parser.add_argument("--space-id", default="LuciferMorningstar777/system-q-demo")
    args = parser.parse_args()
    
    deploy(args.token, args.space_id)
