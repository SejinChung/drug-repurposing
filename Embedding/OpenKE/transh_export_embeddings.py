import os
from datetime import datetime

from openke.module.model import TransE
from openke.data import TrainDataLoader


def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def main():
    checkpoint_path = "./checkpoint/transh/transh_2026-01-05_15-26-59.ckpt" # checkpoint path
    current_time = now_str()

    os.makedirs("./embeddings", exist_ok=True)

    # ===== Hyperparameter (TransH) =====
    nbatches = 100
    neg_samples = 25
    embedding_dim = 200
    alpha_value = 1.0
    train_times = 1000
    margin = 5.0

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

    transh = TransH(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=embedding_dim,
        p_norm=1,
        norm_flag=True
    )

    # ===== Load checkpoint =====
    transh.load_checkpoint(checkpoint_path)

    # ===== Extract Embeddings  =====
    # extract entity embeddings
    entity_embeddings = transh.ent_embeddings.weight
    entity_embeddings = entity_embeddings.cpu().detach().numpy()
    with open(f'./embeddings/transh_entity_embeddings_{current_time}.txt','w') as f:
        for idx, embedding in enumerate(entity_embeddings):
            f.write(f"Entity_{idx}: {embedding.tolist()}\n")
    print(f"Entity embeddings matrix shape: {entity_embeddings.shape}")

    # extract relation embeddings
    relation_embeddings = transh.rel_embeddings.weight
    relation_embeddings = relation_embeddings.cpu().detach().numpy()
    with open(f'./embeddings/transh_relation_embeddings_{current_time}.txt', 'w') as f:
        for idx, embedding in enumerate(relation_embeddings):
            f.write(f"Relation_{idx}: {embedding.tolist()}\n")
    print(f"Relation embeddings matrix shape: {relation_embeddings.shape}")

    # extract norm vector embeddings
    norm_vector_embeddings = transh.norm_vector.weight
    norm_vector_embeddings = norm_vector_embeddings.cpu().detach().numpy()
    with open(f'./embeddings/transh_norm_vector_embeddings_{current_time}.txt', 'w') as f:
        for idx, embedding in enumerate(norm_vector_embeddings):
            f.write(f"Norm_Vector_{idx}: {embedding.tolist()}\n")
    print(f"Norm vector embeddings matrix shape: {norm_vector_embeddings.shape}")


if __name__ == "__main__":
    main()