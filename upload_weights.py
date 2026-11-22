"""Upload trained artifacts to the Hugging Face Hub so the repository stays free of large files.

Usage:  HF_TOKEN=hf_xxx python upload_weights.py <username>/swe015-chat-moderation
"""
import os
import sys

from huggingface_hub import HfApi

from src import config

repo_id = sys.argv[1] if len(sys.argv) > 1 else config.HF_REPO_ID
api = HfApi(token=os.environ.get("HF_TOKEN"))
api.create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
api.upload_folder(repo_id=repo_id, folder_path=str(config.ART_DIR), repo_type="model",
                  ignore_patterns=["bert_fp32/*.bin"])
print(f"Uploaded to https://huggingface.co/{repo_id}")
