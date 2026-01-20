import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def torch_load_any(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # older torch without weights_only
        return torch.load(path, map_location="cpu")


def load_pykeen_from_checkpoint_dir(checkpoint_dir: Path):
    try:
        from pykeen.pipeline import load_pipeline_result
        return load_pipeline_result(checkpoint_dir)
    except Exception:
        pass

    candidates = []
    for name in [
        "pipeline_result.pkl",
        "pipeline_result.pt",
        "result.pkl",
        "trained_model.pkl",
        "trained_model.pt",
        "model.pkl",
        "model.pt",
    ]:
        p = checkpoint_dir / name
        if p.exists():
            candidates.append(p)

    candidates.extend(sorted(checkpoint_dir.glob("*.pkl")))
    candidates.extend(sorted(checkpoint_dir.glob("*.pt")))

    seen = set()
    uniq = []
    for p in candidates:
        if p not in seen:
            uniq.append(p)
            seen.add(p)

    if not uniq:
        raise RuntimeError(
            f"No .pkl/.pt candidates found in checkpoint_dir={checkpoint_dir}\n"
            "Run: dir <checkpoint_dir> and confirm what files exist."
        )

    last_err = None
    for p in uniq:
        try:
            obj = torch_load_any(p)

            if hasattr(obj, "model"):
                return obj

            if isinstance(obj, dict):
                if "model" in obj:
                    return obj["model"]
                if "pipeline_result" in obj:
                    return obj["pipeline_result"]

            if hasattr(obj, "entity_representations") and hasattr(obj, "relation_representations"):
                return obj

        except Exception as e:
            last_err = (p, e)
            continue

    p, e = last_err
    raise RuntimeError(
        "Could not load any candidate file from the checkpoint directory.\n"
        f"Last tried: {p}\n"
        f"Error: {repr(e)}\n"
        "Tip: The directory listing will reveal the correct saved artifact name."
    )


def get_all_entity_embeddings(model) -> np.ndarray:
    rep = model.entity_representations[0]
    num = model.num_entities
    with torch.no_grad():
        idx = torch.arange(num, device=model.device)
        emb = rep(idx)
    return emb.detach().cpu().numpy()


def get_all_relation_embeddings(model) -> np.ndarray:
    rep = model.relation_representations[0]
    num = model.num_relations
    with torch.no_grad():
        idx = torch.arange(num, device=model.device)
        emb = rep(idx)
    return emb.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--out_dir", default="./embeddings")
    args = ap.parse_args()

    ts = now_str()
    checkpoint_dir = Path(args.checkpoint_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    loaded = load_pykeen_from_checkpoint_dir(checkpoint_dir)

    model = loaded.model if hasattr(loaded, "model") else loaded
    model.eval()

    ent = get_all_entity_embeddings(model)
    rel = get_all_relation_embeddings(model)

    ent_path = Path(args.out_dir) / f"rgcn_entity_embeddings_{ts}.txt"
    rel_path = Path(args.out_dir) / f"rgcn_relation_embeddings_{ts}.txt"

    with ent_path.open("w", encoding="utf-8") as f:
        for idx, embedding in enumerate(ent):
            f.write(f"Entity_{idx}: {embedding.tolist()}\n")

    with rel_path.open("w", encoding="utf-8") as f:
        for idx, embedding in enumerate(rel):
            f.write(f"Relation_{idx}: {embedding.tolist()}\n")

    print(f"Entity embeddings matrix shape: {ent.shape}")
    print(f"Relation embeddings matrix shape: {rel.shape}")
    print(f"[Saved] {ent_path}")
    print(f"[Saved] {rel_path}")


if __name__ == "__main__":
    main()
