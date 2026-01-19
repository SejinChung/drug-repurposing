import os
from datetime import datetime

from openke.module.model import TransE, TransR
from openke.data import TrainDataLoader


def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def main():
    checkpoint_path = "./checkpoint/transr/transr_2026-01-05_15-26-59.ckpt" # checkpoint path
    current_time = now_str()

    os.makedirs("./embeddings", exist_ok=True)

    # ===== Hyperparameter (TransE) =====
    nbatches = 200
    neg_samples = 20
    embedding_dim = 50
    alpha_value = 0.5
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

    transr = TransR(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim_e=embedding_dim,
        dim_r=embedding_dim,
        p_norm=1,
        norm_flag=True,
        rand_init=False,
    )

    # ===== Load checkpoint =====
    transr.load_checkpoint(checkpoint_path)

    # ===== Extract Embeddings  =====
    # extract entity embeddings
    entity_embeddings = transr.ent_embeddings.weight
    entity_embeddings = entity_embeddings.cpu().detach().numpy()
    with open(f'./embeddings/transr_entity_embeddings_{current_time}.txt','w') as f:
        for idx, embedding in enumerate(entity_embeddings):
            f.write(f"Entity_{idx}: {embedding.tolist()}\n")
    print(f"Entity embeddings matrix shape: {entity_embeddings.shape}")

    # extract relation embeddings
    relation_embeddings = transr.rel_embeddings.weight
    relation_embeddings = relation_embeddings.cpu().detach().numpy()
    with open(f'./embeddings/transr_relation_embeddings_{current_time}.txt', 'w') as f:
        for idx, embedding in enumerate(relation_embeddings):
            f.write(f"Relation_{idx}: {embedding.tolist()}\n")
    print(f"Relation embeddings matrix shape: {relation_embeddings.shape}")

    # extract matrix embeddings
    matrix_embeddings = transr.transfer_matrix.weight
    matrix_embeddings = matrix_embeddings.cpu().detach().numpy()
    num_relations = matrix_embeddings.shape[0]
    dim_entity = transr.ent_embeddings.embedding_dim
    dim_relation = transr.rel_embeddings.embedding_dim

    with open(f'./embeddings/transr_matrix_embeddings_{current_time}.txt', 'w') as f:
        for idx, embedding in enumerate(matrix_embeddings):
            # reshape to 2D matrix (dim_relation x dim_entity)
            embedding_matrix = embedding.reshape(dim_relation, dim_entity)
            f.write(f"Matrix_{idx}:\n")
            for row in embedding_matrix:
                f.write(f"  {row.tolist()}\n")
            f.write("\n")
    print(f"Matrix embeddings matrix shape: {matrix_embeddings.shape}")

if __name__ == "__main__":
    main()