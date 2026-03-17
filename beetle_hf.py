# =========================================================
# INSTALLS
# =========================================================
#!pip install -q -U huggingface_hub transformers torch tokenizers

# =========================================================
# IMPORTS
# =========================================================
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
import os, re, json, tempfile, time
from collections import defaultdict

api = HfApi()
print("✅ Using HuggingFace API")

# =========================================================
# MODEL LIST
# =========================================================
ALL_REPOS = [
    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_part_time",
    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_part_time",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_part_time",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_part_time",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_part_time",
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_part_time",

    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_late",
    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_late",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_late",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_late",
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_late",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_late",

    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_heritage",
    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_heritage",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_heritage",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_heritage",
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_heritage",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_heritage",

    "BeetleLM/beetlelm_zho_L1-eng_L2-nld_L3_balanced",
    "BeetleLM/beetlelm_zho_L1-nld_L2-eng_L3_balanced",
    "BeetleLM/beetlelm_nld_L1-zho_L2-eng_L3_balanced",
    "BeetleLM/beetlelm_nld_L1-eng_L2-zho_L3_balanced",
    "BeetleLM/beetlelm_eng_L1-nld_L2-zho_L3_balanced",
    "BeetleLM/beetlelm_eng_L1-zho_L2-nld_L3_balanced",

    "BeetleLM/beetlelm_ukr_mono",
    "BeetleLM/beetlelm_ind_mono",
    "BeetleLM/beetlelm_fra_mono",
    "BeetleLM/beetlelm_fas_mono",
    "BeetleLM/beetlelm_deu_mono",
    "BeetleLM/beetlelm_bul_mono",

    "BeetleLM/beetlelm_zho-ukr_simultaneous",
    "BeetleLM/beetlelm_nld-ukr_simultaneous",
    "BeetleLM/beetlelm_zho-fas_simultaneous",
    "BeetleLM/beetlelm_fra-ind_simultaneous",
    "BeetleLM/beetlelm_nld-bul_simultaneous",
    "BeetleLM/beetlelm_zho-ind_simultaneous",
    "BeetleLM/beetlelm_fas-ind_simultaneous",
    "BeetleLM/beetlelm_fas-bul_simultaneous",
    "BeetleLM/beetlelm_zho-deu_simultaneous",
    "BeetleLM/beetlelm_nld-fra_simultaneous",
    "BeetleLM/beetlelm_bul-ukr_simultaneous",
    "BeetleLM/beetlelm_eng-fra_simultaneous",
    "BeetleLM/beetlelm_ind-deu_simultaneous",
    "BeetleLM/beetlelm_fas-ukr_simultaneous",
    "BeetleLM/beetlelm_zho-bul_simultaneous",
    "BeetleLM/beetlelm_zho-fra_simultaneous",
    "BeetleLM/beetlelm_fas-eng_simultaneous",
    "BeetleLM/beetlelm_nld-deu_simultaneous",
    "BeetleLM/beetlelm_fra-ukr_simultaneous",

    "BeetleLM/beetlelm_zho-fas_sequential",
    "BeetleLM/beetlelm_zho-ukr_sequential",
    "BeetleLM/beetlelm_zho-ind_sequential",
    "BeetleLM/beetlelm_zho-deu_sequential",
    "BeetleLM/beetlelm_nld-bul_sequential",
    "BeetleLM/beetlelm_nld-ukr_sequential",
    "BeetleLM/beetlelm_zho-fra_sequential",
    "BeetleLM/beetlelm_fas-ind_sequential",
    "BeetleLM/beetlelm_fra-ind_sequential",
    "BeetleLM/beetlelm_ind-deu_sequential",
    "BeetleLM/beetlelm_nld-fra_sequential",
    "BeetleLM/beetlelm_fas-bul_sequential",
    "BeetleLM/beetlelm_eng-fra_sequential",
    "BeetleLM/beetlelm_fas-ukr_sequential",
    "BeetleLM/beetlelm_fas-eng_sequential",
    "BeetleLM/beetlelm_nld-deu_sequential",
    "BeetleLM/beetlelm_zho-bul_sequential",
    "BeetleLM/beetlelm_eng-nld_sequential",
    "BeetleLM/beetlelm_eng-bul_sequential",

    "BeetleLM/beetlelm_zho-ukr_part_time",
    "BeetleLM/beetlelm_zho-ind_part_time",
    "BeetleLM/beetlelm_nld-ukr_part_time",
    "BeetleLM/beetlelm_zho-fas_part_time",
    "BeetleLM/beetlelm_nld-fra_part_time",
    "BeetleLM/beetlelm_zho-deu_part_time",
    "BeetleLM/beetlelm_fra-ind_part_time",
    "BeetleLM/beetlelm_nld-bul_part_time",
    "BeetleLM/beetlelm_ind-deu_part_time",
    "BeetleLM/beetlelm_fas-ukr_part_time",
    "BeetleLM/beetlelm_zho-bul_part_time",
    "BeetleLM/beetlelm_zho-fra_part_time",
    "BeetleLM/beetlelm_fas-ind_part_time",
    "BeetleLM/beetlelm_fas-bul_part_time",
    "BeetleLM/beetlelm_fas-eng_part_time",
    "BeetleLM/beetlelm_eng-nld_part_time",
    "BeetleLM/beetlelm_fra-ukr_part_time",
    "BeetleLM/beetlelm_nld-deu_part_time",
    "BeetleLM/beetlelm_eng-fra_part_time",
    "BeetleLM/beetlelm_bul-ukr_part_time",
    "BeetleLM/beetlelm_fas-nld_part_time",
    "BeetleLM/beetlelm_fas-deu_part_time",
    "BeetleLM/beetlelm_bul-fra_part_time",
    "BeetleLM/beetlelm_eng-bul_part_time",
    "BeetleLM/beetlelm_deu-ukr_part_time",
    "BeetleLM/beetlelm_eng-ind_part_time",
]

# =========================================================
# WRITE pico_decoder.py
# =========================================================
PICO_PATH = "pico_decoder.py"

# (same content as yours — unchanged)
# 👉 KEEP EXACT FILE (omitted here for brevity in explanation)
# 👉 paste your pico_decoder.py content here exactly

# =========================================================
# CONFIG TEMPLATE
# =========================================================
BASE_CONFIG = {
    "architectures": ["PicoDecoderHF"],
    "model_type": "pico_decoder",
    "auto_map": {
        "AutoConfig": "pico_decoder.PicoDecoderHFConfig",
        "AutoModelForCausalLM": "pico_decoder.PicoDecoderHF",
    },
    "n_layers": 14,
    "d_model": 768,
    "attention_n_heads": 12,
    "attention_n_kv_heads": 1,
    "max_seq_len": 512,
    "batch_size": 64,
    "position_emb_theta": 10000.0,
    "activation_hidden_dim": 3072,
    "norm_eps": 1e-5,
    "dropout": 0.1,
    "torch_dtype": "float32",
}

# =========================================================
# GET VOCAB SIZE
# =========================================================
def get_vocab(repo):
    branches = [b.name for b in api.list_repo_refs(repo).branches]
    steps = sorted([b for b in branches if re.match(r"step-\d+", b)],
                   key=lambda x: int(x.split("-")[1]))

    for b in steps:
        try:
            path = hf_hub_download(repo, "tokenizer.json", revision=b)
            data = json.load(open(path))
            vocab = data.get("model", {}).get("vocab", {})
            if vocab:
                return len(vocab)
        except:
            continue
    return None

# =========================================================
# PUSH FUNCTION (with retry + rate limit handling)
# =========================================================
def push_repo(repo):
    print(f"\n🚀 {repo}")

    vocab = get_vocab(repo)
    if not vocab:
        print("  ❌ no vocab"); return

    config = {**BASE_CONFIG, "vocab_size": vocab}

    branches = [b.name for b in api.list_repo_refs(repo).branches]
    steps = sorted([b for b in branches if re.match(r"step-\d+", b)],
                   key=lambda x: int(x.split("-")[1]))

    if "main" in branches:
        steps.append("main")

    for b in steps:
        success = False
        while not success:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    cp = os.path.join(tmp, "config.json")
                    with open(cp, "w") as f:
                        json.dump(config, f, indent=2)

                    api.create_commit(
                        repo_id=repo,
                        repo_type="model",
                        operations=[
                            CommitOperationAdd("pico_decoder.py", PICO_PATH),
                            CommitOperationAdd("config.json", cp),
                        ],
                        commit_message=f"Fix HF compatibility (vocab={vocab})",
                        revision=b,
                    )

                print(f"  ✅ {b}")
                success = True

                time.sleep(1)

            except Exception as e:
                msg = str(e)

                if "429" in msg or "rate" in msg.lower():
                    print(f"  ⏳ rate limit → sleeping 60s")
                    time.sleep(60)
                elif "No files have been modified" in msg:
                    print(f"  ⏭ {b} (unchanged)")
                    success = True
                else:
                    print(f"  ❌ {b}: {msg[:80]}")
                    success = True  # skip hard failures

# =========================================================
# RUN ALL
# =========================================================
for repo in ALL_REPOS:
    push_repo(repo)

print("\n🎉 DONE — all models processed")
