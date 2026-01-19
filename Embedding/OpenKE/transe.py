import os, json, time, platform
from datetime import datetime

import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader


def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def cuda_reset_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

def cuda_peak_mb():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024**2), 2),
        }
    except Exception as e:
        return {"error": str(e)}

def cuda_device_info():
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        return {
            "cuda_available": True,
            "gpu_name": props.name,
            "gpu_total_mem_gb": round(props.total_memory / (1024**3), 3),
        }
    except Exception as e:
        return {"cuda_available": None, "error": str(e)}

def main():
    # ===== datetime & paths =====
    current_time = now_str()
    os.makedirs("./checkpoint/transe", exist_ok=True)
    os.makedirs("./reports", exist_ok=True)

    checkpoint_path = f'./checkpoint/transe/transe_{current_time}.ckpt'
    report_path = f'./reports/transe_cost_report_{current_time}.json'

    # ===== Hyperparameter (TransE) =====
    nbatches = 50
    neg_samples = 25
    embedding_dim = 200
    alpha_value = 1.0
    train_times = 1000
    margin = 5.0

    # ===== Train & Test loaders =====
    t0 = time.perf_counter()
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/all/",
        nbatches=nbatches,     
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=neg_samples,
        neg_rel=0
    )
    test_dataloader = TestDataLoader("./benchmarks/all/", "link")
    t1 = time.perf_counter()
    dataloader_init_time = t1 - t0

    # ===== Model =====
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=embedding_dim,
        p_norm=1,
        norm_flag=True
    )

    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=margin),
        batch_size=train_dataloader.get_batch_size()
    )

    # ===== Report skeleton =====
    report = {
        "timestamp": current_time,
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "data": {
            "in_path": "./benchmarks/all/",
            "ent_tot": int(train_dataloader.get_ent_tot()),
            "rel_tot": int(train_dataloader.get_rel_tot()),
            "batch_size_openke": int(train_dataloader.get_batch_size()),
            "neg_ent": int(neg_samples),
            "neg_rel": 0,
        },
        "hyperparams": {
            "nbatches": nbatches,
            "embedding_dim": embedding_dim,
            "alpha": alpha_value,
            "train_times": train_times,
            "margin": margin,
            "p_norm": 1,
            "norm_flag": True,
            "threads": 8,
            "bern_flag": 1,
            "filter_flag": 1,
            "sampling_mode": "normal",
        },
        "timing_sec": {
            "dataloader_init": dataloader_init_time,
        },
        "gpu": {
            "device": cuda_device_info(),
        },
        "results": {},
        "artifacts": {
            "checkpoint_path": checkpoint_path,
        }
    }

    # ===== Train =====
    cuda_reset_peak()
    t0 = time.perf_counter()
    trainer = Trainer(
        model=model,
        data_loader=train_dataloader,
        train_times=train_times,
        alpha=alpha_value,
        use_gpu=True
    )
    trainer.run()
    t1 = time.perf_counter()
    report["timing_sec"]["train_wall"] = t1 - t0
    report["gpu"]["train_peak"] = cuda_peak_mb()

    # ===== Save checkpoint =====
    t0 = time.perf_counter()
    transe.save_checkpoint(checkpoint_path)
    t1 = time.perf_counter()
    report["timing_sec"]["save_checkpoint"] = t1 - t0

    # ===== Test =====
    t0 = time.perf_counter()
    transe.load_checkpoint(checkpoint_path)
    tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
    mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=False)
    t1 = time.perf_counter()

    report["timing_sec"]["eval_wall"] = t1 - t0
    report["results"].update({
        "MRR": float(mrr),
        "MR": float(mr),
        "Hit@10": float(hit10),
        "Hit@3": float(hit3),
        "Hit@1": float(hit1),
    })

    # ===== Save report =====
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Saved checkpoint:", checkpoint_path)
    print("Saved report:", report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()