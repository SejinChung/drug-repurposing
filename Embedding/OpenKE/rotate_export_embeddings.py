import os
from datetime import datetime

from openke.module.model import RotatE
from openke.data import TrainDataLoader


def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def main():
    checkpoint_path = "./checkpoint/rotate/rotate_2026-01-05_15-26-59.ckpt" # checkpoint path
    current_time = now_str()

    os.makedirs("./embeddings", exist_ok=True)

    # ===== Hyperparameter (RotatE) =====
    nbatches = 100
    neg_samples = 10
    embedding_dim = 50
    alpha_value = 5e-5
    train_times = 1000

    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/all/",
        nbatches=nbatches,
        threads=8,
        sampling_mode="cross",
        bern_flag=0,
        filter_flag=1,
        neg_ent=neg_samples,
        neg_rel=0
    )

    rotate = RotatE(
	    ent_tot = train_dataloader.get_ent_tot(),
	    rel_tot = train_dataloader.get_rel_tot(),
	    dim = embedding_dim, 
	    margin = 6.0,
	    epsilon = 2.0,
    )

    # ===== Load checkpoint =====
    rotate.load_checkpoint(checkpoint_path)

    # ===== Extract Embeddings  =====
    # extract entity embeddings
    entity_embeddings = rotate.ent_embeddings.weight
    entity_embeddings = entity_embeddings.cpu().detach().numpy()
    with open(f'./embeddings/rotate_entity_embeddings_{current_time}.txt', 'w') as f:
        for idx, embedding in enumerate(entity_embeddings):
            f.write(f"Entity_{idx}: {embedding.tolist()}\n")
    print(f"Entity embeddings matrix shape: {entity_embeddings.shape}")

    # extract relation embeddings
    relation_embeddings = rotate.rel_embeddings.weight
    relation_embeddings = relation_embeddings.cpu().detach().numpy()
    with open(f'./embeddings/rotate_relation_embeddings_{current_time}.txt', 'w') as f:
        for idx, embedding in enumerate(relation_embeddings):
            f.write(f"Relation_{idx}: {embedding.tolist()}\n")
    print(f"Relation embeddings matrix shape: {relation_embeddings.shape}")


if __name__ == "__main__":
    main()