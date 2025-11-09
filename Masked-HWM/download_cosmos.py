from huggingface_hub import login, snapshot_download
import os

login(token="<token>", add_to_git_credential=True)
model_names = [
    "Cosmos-0.1-Tokenizer-DV8x8x8",
]
for model_name in model_names:
    hf_repo = "nvidia/" + model_name
    local_dir = "pretrained_ckpts/" + model_name
    os.makedirs(local_dir, exist_ok=True)
    print(f"downloading {model_name}...")
    snapshot_download(repo_id=hf_repo, local_dir=local_dir)