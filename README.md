# ISL-drug-repurposing
Ensemble-Based Multi-Hop Knowledge Graph for Drug Repurposing in Type 2 Diabetes and Its Comorbidities

## Source
We use **OpenBioLink2020**,(directed, high-quality version) benchmark dataset for biomedical knowledge graph learning. The dataset is publicly available at: https://zenodo.org/records/3834052

## Embedding Models
### 1. OpenKE
We employ classical knowledge graph embedding models implemented in **OpenKE**(https://github.com/thunlp/OpenKE), including **TransE**, **TransH**, **TransR**, **ComplEx**, and **RotatE**.

### 2. R-GCN
Relational Graph Convolutional Networks(R-GCN) are implemented using the **PyKEEN** library.

### 3. HittER
We utilized the **HittER** model provided by **LibKGE**(https://github.com/uma-pi1/kge), for Transformer-based reasoning.
