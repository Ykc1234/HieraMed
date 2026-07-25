# HieraMed

Official PyTorch implementation of **HieraMed: A Hierarchical Representation Framework for Next-Visit Diagnosis Prediction from Longitudinal Electronic Health Records**.

HieraMed formulates next-visit diagnosis prediction as a multi-label learning problem and organizes longitudinal EHR information at three levels:

1. **Code level:** medical-code descriptions are encoded with Sentence-BERT to obtain semantic representations of diagnoses, procedures, and medications.
2. **Visit level:** each encounter is represented as a visit-centered heterogeneous graph, and relation-aware attention aggregates heterogeneous clinical events.
3. **Patient level:** a Liquid Neural Network models the evolution of patient states across longitudinal visits.

The method is evaluated on **MIMIC-III** and **MIMIC-IV**.

## Repository structure

```text
HieraMed/
├── chatgpt_desc/          # Generated descriptions and precomputed semantic embeddings
├── data/                  # Task definitions, preprocessing, and graph construction
├── layers/                # Relation-aware graph and temporal encoding layers
├── models/                # HieraMed and baseline model implementations
├── networks/              # Liquid Neural Network components
├── train.py               # Training and evaluation entry point
├── utils.py               # Data loading, metrics, and training utilities
└── Explain.py             # Optional embedding analysis and visualization
```

## Requirements

The experiments were developed with the following core environment:

- Python 3.10.13
- PyTorch 1.12.1
- PyTorch Geometric 2.3.1
- PyHealth 1.1.4

Additional packages include `torch-sparse`, `numpy`, `scipy`, `scikit-learn`, `pandas`, `joblib`, `dill`, `tqdm`, `matplotlib`, `seaborn`, and `umap-learn`.

Install PyTorch and PyTorch Geometric using versions compatible with your CUDA environment, and then install the remaining dependencies.

## Data access and preprocessing

MIMIC-III and MIMIC-IV are publicly available through PhysioNet after completion of the required credentialing and data-use procedures:

- MIMIC-III: <https://physionet.org/content/mimiciii/>
- MIMIC-IV: <https://physionet.org/content/mimiciv/>

Raw MIMIC files are not distributed in this repository.

The preprocessing protocol follows the manuscript:

- retain patients with at least two recorded visits;
- map diagnosis codes to CCS categories;
- represent medications using third-level ATC codes;
- use grouped ICD-9/10 procedure codes;
- construct next-visit samples from preceding longitudinal history;
- split the data by patient into 75% training, 10% validation, and 15% test sets.

Place the datasets in local directories and update `fileroot` in `train.py` when necessary:

```text
HieraMed/
├── mimic3/
└── mimic4/
```

> **Important:** The reported experiments use patient-level splits. Ensure that no patient appears in more than one data partition.

## Medical-code descriptions and embeddings

HieraMed uses GPT-4o only as an **offline description-generation tool**. No language-model API call is required during model training or evaluation.

The prompt template used in the manuscript is:

```text
Give me a brief explanation for the {medical entity type} using the {coding system} code: {code}.
```

Descriptions were generated with deterministic decoding (`temperature = 0`), encoded using a pretrained Sentence-BERT model, and fixed before downstream training. The `chatgpt_desc/` directory contains the description and embedding files used by the implementation.

Before running the code, verify that `chatgpt_desc/load_embedding.py` resolves these files using repository-relative paths on your system.

## Training HieraMed

Create the checkpoint directory:

```bash
mkdir -p logs
```

Train on MIMIC-III:

```bash
python train.py \
  --model HieraMed \
  --dataset mimic3 \
  --epochs 500 \
  --lr 0.001 \
  --batch_size 256 \
  --unfolding_steps 5 \
  --seed 42 \
  --is_train True
```

Train on MIMIC-IV:

```bash
python train.py \
  --model HieraMed \
  --dataset mimic4 \
  --epochs 500 \
  --lr 0.001 \
  --batch_size 256 \
  --unfolding_steps 5 \
  --seed 42 \
  --is_train True
```

The manuscript selects hyperparameters using validation Precision@20 from the following search space:

- learning rate: `{0.01, 0.005, 0.001}`;
- dropout: `{0.0, 0.1, ..., 0.6}`;
- hidden dimension: `{32, 64, 128}`.

Each reported experiment is repeated using 10 random seeds.

## Evaluation

After training, run the same command without enabling training:

```bash
python train.py --model HieraMed --dataset mimic3 --seed 42
```

or

```bash
python train.py --model HieraMed --dataset mimic4 --seed 42
```

Checkpoints are stored under:

```text
logs/trained_<MODEL>_<DATASET>.ckpt
```

The implementation reports the manuscript-defined metrics at `k = 10, 20, 30`:

- visit-level Precision@k;
- code-level Accuracy@k.

## Baselines

The repository contains implementations or wrappers for several comparison methods, including:

- Transformer
- RETAIN
- StageNet
- KAME
- TRANS

Example:

```bash
python train.py --model RETAIN --dataset mimic3 --epochs 500 --is_train True
```

The manuscript additionally reports comparisons with GCT, HiTANet, and DDHGNN under the same preprocessing and evaluation protocol.

## Main results reported in the manuscript

| Dataset | Visit-level Precision@20 | Code-level Accuracy@20 |
|---|---:|---:|
| MIMIC-III | 67.80% | 63.39% |
| MIMIC-IV | 71.65% | 65.64% |

Values are averaged over 10 runs with different random seeds.


## Citation

If you use this repository, please cite:

> Feifei Ke, Deli Hua, and Kaichao Yang. **HieraMed: A Hierarchical Representation Framework for Next-Visit Diagnosis Prediction from Longitudinal Electronic Health Records.**

The complete bibliographic record will be added after publication.

## Contact

For questions about the method or implementation, please contact the corresponding author:

Kaichao Yang — `m18358880096@163.com`
