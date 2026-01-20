import argparse
from pathlib import Path
from datetime import datetime
import torch


def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def load_state_dict(path: Path):
    print(f"[INFO] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    sd = ckpt["model"][0]
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="./embeddings")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = now_str()

    state_dict = load_state_dict(ckpt_path)

    ent_key = "_entity_embedder._embeddings.weight"
    rel_key = "_relation_embedder._embeddings.weight"

    ent = state_dict[ent_key]
    rel = state_dict[rel_key]

    print(f"[INFO] Entity shape   : {ent.shape}")
    print(f"[INFO] Relation shape : {rel.shape}")

    ent_path = out_dir / f"hitter_entity_embeddings_{ts}.txt"
    rel_path = out_dir / f"hitter_relation_embeddings_{ts}.txt"

    # ===== save =====
    with ent_path.open("w") as f:
        for i, vec in enumerate(ent):
            f.write(f"Entity_{i}: {vec.tolist()}\n")

    with rel_path.open("w") as f:
        for i, vec in enumerate(rel):
            f.write(f"Relation_{i}: {vec.tolist()}\n")

    print("\n[SAVED]")
    print(ent_path)
    print(rel_path)


if __name__ == "__main__":
    main()
