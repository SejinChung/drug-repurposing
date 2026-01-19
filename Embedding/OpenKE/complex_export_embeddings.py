import os
from datetime import datetime

from openke.module.model import ComplEx
from openke.data import TrainDataLoader


def now_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def main():
    checkpoint_path = "./checkpoint/complex/complex_2026-01-05_15-26-59.ckpt" # checkpoint path
    current_time = now_str()

    os.makedirs("./embeddings", exist_ok=True)

    # ===== Hyperparameter (ComplEx) =====
    nbatches = 200
    neg_samples = 10
    embedding_dim = 100
    alpha_value = 0.5
    train_times = 1000

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

    complEx = ComplEx(
	    ent_tot = train_dataloader.get_ent_tot(),
	    rel_tot = train_dataloader.get_rel_tot(),
	    dim = embedding_dim
    )

    # ===== Load checkpoint =====
    complEx.load_checkpoint(checkpoint_path)

    # ===== Extract Embeddings  =====
    # extract entity embeddings (im)
    entity_im_embeddings = complEx.ent_im_embeddings.weight
    entity_im_embeddings = entity_im_embeddings.cpu().detach().numpy()
    with open(f'./embeddings/complex_entity_im_embeddings_{current_time}.txt','w') as f:
        for idx, embedding in enumerate(entity_im_embeddings):
            f.write(f"Entity_{idx}: {embedding.tolist()}\n")
    print(f"Entity embeddings matrix shape: {entity_im_embeddings.shape}")

    # extract entity embeddings (re)
    entity_re_embeddings = complEx.ent_re_embeddings.weight
    entity_re_embeddings = entity_re_embeddings.cpu().detach().numpy()
    with open(f'./embeddings/complex_entity_re_embeddings_{current_time}.txt','w') as f:
        for idx, embedding in enumerate(entity_re_embeddings):
            f.write(f"Entity_{idx}: {embedding.tolist()}\n")
    print(f"Entity embeddings matrix shape: {entity_re_embeddings.shape}")

    # extract relation embeddings (im)
    relation_im_embeddings = complEx.rel_im_embeddings.weight
    relation_im_embeddings = relation_im_embeddings.cpu().detach().numpy()
    with open(f'./embeddings/complex_relation_im_embeddings_{current_time}.txt','w') as f:
        for idx, embedding in enumerate(relation_im_embeddings):
            f.write(f"Relation_{idx}: {embedding.tolist()}\n")
    print(f"Relation embeddings matrix shape: {relation_im_embeddings.shape}")

    # extract relation embeddings (re)
    relation_re_embeddings = complEx.rel_re_embeddings.weight
    relation_re_embeddings = relation_re_embeddings.cpu().detach().numpy()
    with open(f'./embeddings/complex_relation_re_embeddings_{current_time}.txt','w') as f:
        for idx, embedding in enumerate(relation_re_embeddings):
            f.write(f"Relation_{idx}: {embedding.tolist()}\n")
    print(f"Relation embeddings matrix shape: {relation_re_embeddings.shape}")


if __name__ == "__main__":
    main()