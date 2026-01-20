import argparse
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import torch

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# =========== Utils ==========
def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def read_openke_mapping(path: Path) -> Dict[int, str]:
    # entity2id.txt / relation2id.txt -> {int_id: label_str}
    id_to_label = {}
    with path.open("r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            if not line.strip():
                continue
            *name, idx = line.strip().split()
            id_to_label[int(idx)] = " ".join(name)
    return id_to_label

def read_openke_triples(path: Path) -> List[Tuple[int, int, int]]:
    # train2id.txt / valid2id.txt / test2id.txt
    triples = []
    with path.open("r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            if not line.strip():
                continue
            h, t, r = map(int, line.split())
            triples.append((h, r, t))
    return triples

def to_labeled(triples, ent_map, rel_map):
    # (h_id, r_id, t_id) -> (h_label, r_label, t_label)
    arr = np.empty((len(triples), 3), dtype=object)
    for i, (h, r, t) in enumerate(triples):
        arr[i, 0] = ent_map[h]
        arr[i, 1] = rel_map[r]
        arr[i, 2] = ent_map[t]
    return arr

def filter_labeled_triples_by_vocab(
    triples_lab: np.ndarray,
    ent_vocab: Set[str],
    rel_vocab: Set[str],
) -> np.ndarray:
    keep = []
    for h, r, t in triples_lab:
        if (h in ent_vocab) and (t in ent_vocab) and (r in rel_vocab):
            keep.append((h, r, t))
    return np.asarray(keep, dtype=object)

def get_gpu_snapshot():
    out = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        out["gpu_name"] = props.name
        out["gpu_total_mem_gb"] = round(props.total_memory / (1024**3), 3)
    return out

# ========== main ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--num_batches", type=int, default=200)
    ap.add_argument("--embedding_dim", type=int, default=200)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")  # auto / cpu / cuda:0
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="./runs_rgcn")
    args = ap.parse_args()

    ts = now_str()
    out_dir = Path(args.out_dir) / f"rgcn_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)

    # Report structure
    report = {
        "timestamp": ts,
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "data": {
            "in_path": str(data_dir),
        },
        "hyperparams": {
            "model": "RGCN",
            "epochs": args.epochs,
            "target_num_batches": args.num_batches,
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "seed": args.seed,
            "filtered_eval": True,
        },
        "timing_sec": {},
        "gpu": {},
        "results": {},
        "artifacts": {},
    }

    # Data Load + PyKEEN TriplesFactory
    t0 = time.perf_counter()
    ent_map = read_openke_mapping(data_dir / "entity2id.txt")
    rel_map = read_openke_mapping(data_dir / "relation2id.txt")

    train = read_openke_triples(data_dir / "train2id.txt")
    valid = read_openke_triples(data_dir / "valid2id.txt")
    test  = read_openke_triples(data_dir / "test2id.txt")
    
    train_lab = to_labeled(train, ent_map, rel_map)
    valid_lab = to_labeled(valid, ent_map, rel_map)
    test_lab  = to_labeled(test,  ent_map, rel_map)

    training_tf = TriplesFactory.from_labeled_triples(train_lab)

    ent_vocab = set(training_tf.entity_to_id.keys())
    rel_vocab = set(training_tf.relation_to_id.keys())

    valid_lab_f = filter_labeled_triples_by_vocab(valid_lab, ent_vocab, rel_vocab)
    test_lab_f  = filter_labeled_triples_by_vocab(test_lab,  ent_vocab, rel_vocab)

    validation_tf = TriplesFactory.from_labeled_triples(
        valid_lab_f,
        entity_to_id=training_tf.entity_to_id,
        relation_to_id=training_tf.relation_to_id,
    )
    testing_tf = TriplesFactory.from_labeled_triples(
        test_lab_f,
        entity_to_id=training_tf.entity_to_id,
        relation_to_id=training_tf.relation_to_id,
    )

    n_train = int(training_tf.num_triples)
    batch_size = max(1, math.ceil(n_train / args.num_batches))

    report["timing_sec"]["dataloader_init"] = time.perf_counter() - t0
    report["data"].update({
        "num_entities": training_tf.num_entities,
        "num_relations": training_tf.num_relations,
        "num_train_triples": int(training_tf.num_triples),
        "num_valid_triples": 0, #int(validation_tf.num_triples),
        "num_test_triples": int(testing_tf.num_triples),
        "batch_size": int(batch_size),
    })

    # GPU
    report["gpu"]["device"] = get_gpu_snapshot()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Train + Eval
    device = None if args.device == "auto" else args.device
    t1 = time.perf_counter()
    
    result = pipeline(
        training=training_tf,
        validation=validation_tf,
        testing=testing_tf,
        model="RGCN",
        model_kwargs=dict(
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            edge_dropout=args.dropout,
            self_loop_dropout=args.dropout),
        training_kwargs=dict(
            num_epochs=args.epochs,
            batch_size=batch_size,
            sampler="schlichtkrull"
        ),
        optimizer_kwargs=dict(lr=args.lr),
        evaluator="RankBasedEvaluator",
        evaluator_kwargs=dict(filtered=True),
        stopper="early",
        stopper_kwargs=dict(
            frequency=5,
            patience=5,
            metric="inverse_harmonic_mean_rank",
            larger_is_better=True,
        ),
        device=device,
        random_seed=args.seed,
    )

    report["timing_sec"]["train_eval_wall"] = time.perf_counter() - t1

    if torch.cuda.is_available():
        report["gpu"]["train_peak"] = {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 2),
        }

    # Metrics
    md = result.metric_results.to_flat_dict()
    report["results"] = {
        "MRR": float(md.get("both.realistic.inverse_harmonic_mean_rank")),
        "MR": float(md.get("both.realistic.arithmetic_mean_rank")),
        "Hit@1": float(md.get("both.realistic.hits_at_1")),
        "Hit@3": float(md.get("both.realistic.hits_at_3")),
        "Hit@10": float(md.get("both.realistic.hits_at_10")),
    }

    # Save Checkpoint
    ckpt_dir = out_dir / "checkpoint"
    t2 = time.perf_counter()
    result.save_to_directory(ckpt_dir)
    report["timing_sec"]["save_checkpoint"] = time.perf_counter() - t2
    report["artifacts"]["checkpoint_dir"] = str(ckpt_dir)

    report_path = out_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n===== Results (both.realistic) =====")
    for k, v in report["results"].items():
        print(f"{k:>6} : {v:.6f}")
    print(f"\n[Saved] report: {report_path}")
    print(f"[Saved] checkpoint: {ckpt_dir}")


if __name__ == "__main__":
    main()

'''
python main.py `
  --data_dir .\data\all\ `
  --device cuda `
  --epochs 20 `
  --num_batches 200 `
  --embedding_dim 200 `
  --num_layers 2 `
  --dropout 0.2 `
  --lr 1e-3

foreach ($nb in 1e-3,5e-4,1e-4) {
  python main.py `
    --data_dir .\data\all `
    --device cuda `
    --epochs 5 `
    --num_batches 200 `
    --embedding_dim 200 `
    --num_layers 2 `
    --dropout 0.2 `
    --lr $nb
}
'''