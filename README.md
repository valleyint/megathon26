# Automatic Category Suggestion for Wikimedia Commons

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Megathon X](https://img.shields.io/badge/Megathon%20X-Team%2023-purple.svg)]()

An extreme multi-label category suggestion system for **Wikimedia Commons** ($|\mathcal{C}| \gg 10^6$ categories). The system structures inference into a three-stage cascaded retrieve-and-rerank architecture designed to operate within a strict **sub-200ms latency budget** while providing human-auditable visual evidence and supporting zero-retraining continual learning.

---

## Table of Contents

- [1. System Architecture](#1-system-architecture)
- [2. Codebase Structure](#2-codebase-structure)
- [3. Stage 1 Retrieval Engine](#3-stage-1-retrieval-engine)
  - [Mathematical Formulations](#mathematical-formulations)
  - [Zero-Copy Vision Backbone](#zero-copy-vision-backbone)
  - [Scorer 1: Historical $k$-NN Consensus](#scorer-1-historical-k-nn-consensus)
  - [Scorer 2: Zero-Shot Text Prototypes](#scorer-2-zero-shot-text-prototypes)
  - [Reciprocal Rank Fusion (RRF)](#reciprocal-rank-fusion-rrf)
- [4. Data Contracts & Schemas](#4-data-contracts--schemas)
- [5. Installation & Setup](#5-installation--setup)
- [6. CLI & Workflow Guide](#6-cli--workflow-guide)
  - [Step 1: Dataset Preparation](#step-1-dataset-preparation)
  - [Step 2: Precompute Text Prototypes](#step-2-precompute-text-prototypes)
  - [Step 3: Build Historical FAISS Index](#step-3-build-historical-faiss-index)
  - [Step 4: Run Evaluation Benchmark](#step-4-run-evaluation-benchmark)
  - [Step 5: Interactive Query Demo](#step-5-interactive-query-demo)
- [7. Continual Learning Mechanics](#7-continual-learning-mechanics)
- [8. Test Suite](#8-test-suite)
- [9. Project Roadmap](#9-project-roadmap)

---

## 1. System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                       INPUT DATA                                       │
│  • Uploaded Image x                     • Wikitext Metadata m (Title, Description)     │
│  • Camera EXIF (Timestamp)              • Geotags (Latitude, Longitude)                │
└─────────────────────┬───────────────────────────────────────────┬──────────────────────┘
                      │                                           │
                      ▼                                           ▼
┌───────────────────────────────────────────┐ ┌──────────────────────────────────────────┐
│  STAGE 1: DENSE MULTI-SCORER RECALL       │ │  STAGE 2: METADATA & EVENT DISCOVERY     │
│  (Shared SigLIP/CLIP Encoder) [< 50ms]    │ │  (In Progress)                           │
│  • Normalized feature vector z_x          │ │  • Wikitext markup parsing               │
│  • Scorer 1: Historical k-NN consensus    │ │  • Offline KD-tree reverse geocoding     │
│  • Scorer 2: Zero-shot text prototypes    │ │  • LLM proactive event synthesis         │
│  • Scorer 3: ASL supervised linear head   │ │  • Temporal/spatial conflict filter      │
│  • Reciprocal Rank Fusion (RRF)           │ │  • Category DAG ancestor suppression     │
└─────────────────────┬─────────────────────┘ └───────────────────┬──────────────────────┘
                      │ Top-50 Visual Candidates                  │ Top-20 Event Candidates
                      └─────────────────────┬─────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 2: FILTERED CANDIDATE POOL (10–15 Categories)                │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   STAGE 3: FINE-GRAINED VLM GROUNDING & RERANKING [< 60ms]             │
│  • Single-turn Florence-2 / PaliGemma with grammar-constrained JSON decoding           │
│  • Sibling leaf visual disambiguation & visual grounding check                         │
│  • Calibrated score synthesis: s_final = α · s_prior + β · s_VLM                       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               FINAL SUGGESTION OUTPUT                                  │
│  • Ranked Top-K Wikimedia Commons Categories Ŷ_k(x, m)                                 │
│  • Visual Evidence Phrases & Nearest-Neighbor Exemplars for Human Review               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Codebase Structure

```text
megathon/
├── configs/
│   └── stage1_default.yaml          # Hyperparameters (tau, pool_size, SigLIP weights, paths)
│
├── src/
│   ├── common/                      # Shared types, logging, and profiling
│   │   ├── types.py                 # Core schemas (CategoryInfo, ScoredCandidate, Stage1CandidatePool)
│   │   ├── config.py                # YAML configuration loader
│   │   └── timer.py                 # PipelineTimer for sub-millisecond stage latency measurement
│   │
│   ├── data/                        # Category management & data ingest
│   │   ├── category_vocab.py        # CategoryVocabulary ID <-> name mapping & JSON serialization
│   │   └── commons_dataset.py       # CommonsDataset loader & synthetic dataset generator
│   │
│   ├── encoders/                    # Vision and Text backbones
│   │   ├── base.py                  # Abstract BaseVisionEncoder & BaseTextEncoder
│   │   └── siglip_encoder.py        # SigLIPVisionEncoder & SigLIPTextEncoder (+ mock mode)
│   │
│   ├── stage1/                      # Stage 1 Candidate Generation Engine
│   │   ├── scorers/
│   │   │   ├── base.py              # BaseScorer abstract interface
│   │   │   ├── knn_scorer.py        # Scorer 1: FAISS IVF-PQ / Flat-IP + Consensus Voting
│   │   │   └── prototype_scorer.py  # Scorer 2: Zero-shot prototype dot-product matching
│   │   ├── aggregator.py            # Stage1Aggregator with Reciprocal Rank Fusion (RRF)
│   │   └── pipeline.py              # Stage1Pipeline orchestrator
│   │
│   ├── evaluation/                  # Metric computation & benchmarking
│   │   └── metrics.py               # Precision@k, Recall@k, F1@k, mAP, and latency profiling
│   │
│   └── index_builder/               # Offline tools
│       ├── build_prototypes.py      # Category text prototype matrix builder
│       └── build_faiss.py           # Historical image FAISS index builder
│
├── scripts/                         # Command-Line Interfaces
│   ├── download_dataset.py          # Download or generate test datasets
│   ├── build_prototypes.py          # CLI to generate category prototype embeddings
│   ├── build_faiss_index.py         # CLI to construct FAISS historical index
│   ├── evaluate_stage1.py           # CLI benchmark runner evaluating P@k, R@k, mAP
│   └── demo_stage1.py               # Single-image query CLI tool with latency audit
│
├── tests/                           # Unit & Integration Tests
│   ├── test_category_vocab.py       # Tests vocabulary manipulation and serialization
│   ├── test_metrics.py              # Tests ranking evaluation metrics
│   ├── test_scorers.py              # Tests KNNScorer and PrototypeScorer
│   ├── test_aggregator.py           # Tests Reciprocal Rank Fusion & pool sizing
│   └── test_pipeline.py             # Integration test for end-to-end Stage 1 pipeline
│
├── requirements.txt                 # Python dependencies
├── implentation.txt                 # System specification and design document
└── report1/                         # Milestone 1 (M1) LaTeX report
```

---

## 3. Stage 1 Retrieval Engine

### Mathematical Formulations

#### 1. Zero-Copy Vision Embedding
A single forward pass computes an $L_2$-normalized feature vector for query image $x$:
$$z_x = \frac{f_\theta(x)}{\|f_\theta(x)\|_2} \in \mathbb{R}^d$$
Using `google/siglip-base-patch16-224`, $d = 768$.

#### 2. Scorer 1: Historical $k$-NN Consensus
Retrieves $K = 50$ nearest neighbors $\mathcal{N}_K(x)$ from historical uploads and weights multi-label category assignments via temperature-scaled softmax consensus:
$$S_{\text{knn}}(c \mid x) = \frac{\sum_{i \in \mathcal{N}_K(x)} \exp(z_x^\top z_i / \tau) \cdot \mathbb{I}[c \in Y_i]}{\sum_{i \in \mathcal{N}_K(x)} \exp(z_x^\top z_i / \tau)}$$
where $\tau = 0.07$ and $Y_i$ is the ground-truth category set for neighbor $i$.

#### 3. Scorer 2: Zero-Shot Text Prototypes
Category titles are passed through the matching text encoder $g_\theta$ using standard Commons prompt templates (`"a Wikimedia Commons photograph categorized under {name}"`), producing prototype matrix $T \in \mathbb{R}^{|\mathcal{C}| \times d}$. The similarity is computed via direct inner product:
$$s_{\text{txt}}(c \mid x) = z_x^\top t_c$$

#### 4. Reciprocal Rank Fusion (RRF)
The aggregator fuses candidate lists from enabled scorers to produce a balanced candidate pool:
$$\text{RRF}(c) = \sum_{m \in \{\text{knn}, \text{proto}\}} \frac{w_m}{k_{\text{rrf}} + \text{rank}_m(c) + 1}$$
with default $k_{\text{rrf}} = 60$ and weights $w_{\text{knn}} = w_{\text{proto}} = 0.5$.

---

## 4. Data Contracts & Schemas

### `ScoredCandidate`
Represents an individual category suggested by a scorer:
```python
@dataclass
class ScoredCandidate:
    category_id: int
    name: str                        # "Category:Gothic architecture"
    score: float                     # Individual or fused confidence score
    source: str                      # "knn", "prototype", or "stage1_fused"
    metadata: Dict[str, Any]         # e.g., {"neighbor_exemplars": ["img_001.jpg"]}
```

### `Stage1CandidatePool`
The standardized DTO passed downstream to Stage 2:
```python
@dataclass
class Stage1CandidatePool:
    query_id: str
    embedding: np.ndarray            # Normalized visual embedding z_x
    candidates: List[ScoredCandidate]# Combined unique Top-50 candidates
    scorer_breakdown: Dict[str, List[ScoredCandidate]]
    latency_ms: Dict[str, float]     # Per-component latency breakdown
```

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Linux (x86_64 or aarch64) or macOS

### Virtual Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies Overview
- `torch`, `transformers`: SigLIP backbone inference
- `faiss-cpu`: High-throughput historical image vector search
- `numpy`, `pillow`: Image processing and vector math
- `pyyaml`, `tqdm`: Configuration and progress tracking

---

## 6. CLI & Workflow Guide

### Step 1: Dataset Preparation
Generate a synthetic dataset for testing or download a real Wikimedia Commons subset:
```bash
# Generate a synthetic dataset (30 categories, 60 images)
python scripts/download_dataset.py --synthetic --num-categories 30 --num-images 60
```
This produces:
- `data/commons_split/dataset.json` (metadata & query records)
- `data/commons_split/category_vocab.json` (category taxonomy mapping)
- `data/commons_split/images/` (synthetic JPEG test images)

### Step 2: Precompute Text Prototypes
Compute zero-shot embeddings for all categories in the vocabulary:
```bash
# With real SigLIP model:
python scripts/build_prototypes.py --vocab data/commons_split/category_vocab.json

# Fast mock mode (offline verification):
python scripts/build_prototypes.py --vocab data/commons_split/category_vocab.json --mock
```

### Step 3: Build Historical FAISS Index
Extract visual features across historical images and build the $k$-NN index:
```bash
# With real SigLIP model:
python scripts/build_faiss_index.py --vocab data/commons_split/category_vocab.json

# Fast mock mode (offline verification):
python scripts/build_faiss_index.py --vocab data/commons_split/category_vocab.json --mock
```

### Step 4: Run Evaluation Benchmark
Evaluate ranking accuracy ($P@k$, $R@k$, $F1@k$, mAP) and latency across validation queries:
```bash
python scripts/evaluate_stage1.py --vocab data/commons_split/category_vocab.json --mock
```
Example Output:
```text
==================================================
STAGE 1 RETRIEVAL EVALUATION RESULTS
==================================================
Evaluated Queries: 50
Mean Latency:      22.45 ms
mAP:               0.8240
--------------------------------------------------
k     | Precision@k  | Recall@k   | F1@k    
--------------------------------------------------
1     | 0.7400       | 0.5821     | 0.6516  
5     | 0.3800       | 0.8920     | 0.5330  
10    | 0.2100       | 0.9412     | 0.3435  
20    | 0.1150       | 0.9850     | 0.2058  
50    | 0.0480       | 1.0000     | 0.0916  
==================================================
```

### Step 5: Interactive Query Demo
Query an individual image through the complete Stage 1 pipeline:
```bash
python scripts/demo_stage1.py --image data/commons_split/images/wiki_img_0000.jpg --mock --top-k 5
```

---

## 7. Continual Learning Mechanics

The system supports real-time ingestion without retraining the neural backbones:

1. **Instant Category Ingestion (`add_category`)**:
   When a new category is registered on Commons, its name is embedded via `SigLIPTextEncoder` and appended to the in-memory prototype matrix $T$ in milliseconds:
   ```python
   new_vec = text_encoder.encode_text("Category:New Landmark")
   prototype_scorer.add_category("Category:New Landmark", new_vec)
   ```
2. **Instant Historical Ingestion (`add_historical_images`)**:
   When a user upload is accepted by Commons administrators, its normalized vector $z_x$ and assigned category tags are appended directly to the FAISS index with zero retraining overhead:
   ```python
   knn_scorer.add_historical_images(vectors=new_z, image_ids=["new_img_1"], category_ids_per_image=[[cat_id]])
   ```

---

## 8. Test Suite

Run the unit and integration tests using Python's standard `unittest` framework:

```bash
# Run all tests
python -m unittest discover tests

# Or run individual test modules
python -m unittest tests/test_category_vocab.py
python -m unittest tests/test_metrics.py
python -m unittest tests/test_scorers.py
python -m unittest tests/test_aggregator.py
python -m unittest tests/test_pipeline.py
```

---

## 9. Project Roadmap

- [x] **Milestone 1 (M1):** System Architecture Specification, Flowchart, LaTeX Proposal ([report1/report.tex](report1/report.tex)).
- [x] **Phase 1 Implementation:**
  - [x] SigLIP Vision & Text Encoders with device dispatch.
  - [x] FAISS $k$-NN consensus scorer with temperature scaling.
  - [x] Zero-shot prototype scorer with continual learning addition.
  - [x] Reciprocal Rank Fusion (RRF) candidate aggregator.
  - [x] End-to-end Stage 1 inference pipeline with sub-millisecond profiling.
  - [x] Complete evaluation metrics suite ($P@k$, $R@k$, $F1@k$, mAP).
  - [x] CLI entrypoints and test suite.
- [ ] **Milestone 2 (M2 - 4:00 AM, 11th Oct):** Stage 1 evaluated on temporal validation split + Progress Report PDF.
- [ ] **Stage 2 Implementation:** Metadata parsing, EXIF chronological validation, KD-tree GPS filtering, proactive LLM event synthesis, and Category DAG ancestor suppression.
- [ ] **Stage 3 Implementation:** Lightweight VLM (Florence-2) visual verification and human audit trail generation.
- [ ] **Final Submission (11:00 AM, 11th Oct):** Complete cascaded pipeline, 3-page final report, demo video, and slides.
