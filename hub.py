"""Weights are NOT stored in git (R7). Share them through the Hugging Face Hub instead.

  python hub.py push     --repo <hf-username>/swe015-chat-moderation   # after training (needs `huggingface-cli login`)
  python hub.py download --repo <hf-username>/swe015-chat-moderation   # on a fresh machine, instead of training
The data splits are included so that evaluation uses exactly the same test set.
"""
import argparse

from huggingface_hub import HfApi, snapshot_download

from src import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "download"])
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()
    if a.action == "push":
        api = HfApi()
        api.create_repo(a.repo, exist_ok=True)
        api.upload_folder(repo_id=a.repo, folder_path=str(C.MODELS_DIR), path_in_repo="models")
        api.upload_folder(repo_id=a.repo, folder_path=str(C.DATA_DIR / "splits"), path_in_repo="data/splits")
        print(f"Uploaded to https://huggingface.co/{a.repo}")
    else:
        snapshot_download(repo_id=a.repo, local_dir=str(C.ROOT), allow_patterns=["models/*", "data/*"])
        print("Downloaded models/ and data/splits/")


if __name__ == "__main__":
    main()
