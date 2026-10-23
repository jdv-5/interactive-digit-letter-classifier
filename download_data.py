from huggingface_hub import snapshot_download

archivo_local = snapshot_download(
    repo_id="marhora/mnist-emnist-letter-digit",
    repo_type='dataset',
    allow_patterns="data/*",
    local_dir="./dataset"
)

print(f'files downloaded in {archivo_local}')