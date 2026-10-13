"""
Upload System Q SOTA Adapter to Hugging Face Hub.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login  (or pass --token <your_hf_token>)

Usage:
    python scripts/upload_to_huggingface.py --repo-id "your-username/system-q-3b-decision"
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo


def upload(repo_id: str, token: str = None):
    api = HfApi(token=token)
    
    print(f"Ensuring repository exists: {repo_id}...")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
    
    local_dir = "models/adapters/system_q_sota_direct_scorer"
    if not os.path.exists(local_dir):
        raise FileNotFoundError(f"Adapter folder not found: {local_dir}")
        
    print(f"Uploading files from {local_dir} to {repo_id}...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload System Q SOTA LoRA Adapter (88% benchmark accuracy on 100 questions)"
    )
    
    print(f"\n✅ Upload Complete! Your model is live at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload System Q to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. username/system-q-3b-decision")
    parser.add_argument("--token", default=None, help="Hugging Face Write Token (optional if logged in via CLI)")
    args = parser.parse_args()
    
    upload(args.repo_id, args.token)
