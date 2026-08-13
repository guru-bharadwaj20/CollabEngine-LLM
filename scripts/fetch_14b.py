from huggingface_hub import hf_hub_download
for i in (1, 2, 3):
    f = f"qwen2.5-14b-instruct-q4_k_m-0000{i}-of-00003.gguf"
    print("fetching", f, flush=True)
    p = hf_hub_download(repo_id="Qwen/Qwen2.5-14B-Instruct-GGUF", filename=f, local_dir="models")
    print("ok", p, flush=True)
print("ALL SHARDS DONE", flush=True)
