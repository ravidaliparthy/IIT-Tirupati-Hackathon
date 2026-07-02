# IIT-Tirupati-Hackathon

**AI/ML Hackathon — Ministry of Panchayati Raj (MoPR), Government of India**
Organised by the **Geospatial Intelligence & Applications Laboratory, IIT
Tirupati Navavishkar I-Hub Foundation**, in partnership with **IIIT
Tirupati** and the **National Informatics Centre (NIC)**.

This repository contains our team's solutions to **both problem statements**
of the Geospatial Intelligence Hackathon, built entirely on drone-survey data
collected under the **[SVAMITVA scheme](https://svamitva.nic.in/)** (Survey
of Villages Abadi and Mapping with Improvised Technology in Village Areas) —
India's national programme to survey and map rural inhabited ("abadi") land
using drone imagery and LiDAR.

| | Problem Statement 1 | Problem Statement 2 |
|---|---|---|
| **Goal** | Extract building/road/water/utility features from drone **orthophotos** | Generate a bare-earth **DTM** from LiDAR point clouds and design an optimised **drainage network** |
| **Input** | RGB drone orthophoto GeoTIFFs + SVAMITVA cadastral shapefiles | Raw `.las`/`.laz` LiDAR point clouds |
| **Core technique** | 3-model semantic segmentation ensemble (SegFormer + Mask2Former + custom dual-attention U-Net++) | PointNet-style regression (CSF ground filtering → `TerrainNet`) + D8 hydrology (`pysheds`) |
| **Output** | 10-class feature raster / GeoPackage (roofs by material, roads, waterbodies, transformers, tanks, wells) | DTM GeoTIFF + drainage-network & waterlogging GeoJSON |
| **Code** | [`hack.ipynb`](./hack.ipynb), [`Hackathon.ipynb`](./Hackathon.ipynb) | [`DTM_OptimizedDrainageGeneration-main/`](./DTM_OptimizedDrainageGeneration-main) |
| **Sample data/outputs** | — | [`SVAMITVA_Data_DTM_MULTI-.../`](./SVAMITVA_Data_DTM_MULTI-20260623T143209Z-3-001) |

---

## Table of contents

- [Why this project exists](#why-this-project-exists)
- [Repository structure](#repository-structure)
- [Problem Statement 1 — Feature Extraction](#problem-statement-1--ai-based-feature-extraction-from-drone-orthophotos)
- [Problem Statement 2 — DTM & Drainage](#problem-statement-2--dtm-generation--optimized-drainage-network)
- [Getting started](#getting-started)
- [Data](#data)
- [Results summary](#results-summary)
- [Tech stack](#tech-stack)
- [Team](#team)
- [Acknowledgements](#acknowledgements)

---

## Why this project exists

Under SVAMITVA, drones have already surveyed **6+ lakh (600,000+) Indian
villages**, producing high-resolution orthophotos and, for many, LiDAR point
clouds. That raw imagery is extremely valuable, but today most of the
downstream work — digitising building footprints, classifying roof
materials, drawing roads and waterbodies, designing drainage systems — is
still done **manually** by GIS technicians, which is slow and doesn't scale
to hundreds of thousands of villages.

Both problem statements in this hackathon ask: **can AI/ML turn the drone
data India has already collected into ready-to-use planning layers,
automatically?**

- **Problem Statement 1** automates feature digitisation from orthophotos
  (what's on the ground: which buildings, what they're made of, where the
  roads/water/infrastructure are).
- **Problem Statement 2** automates terrain modelling and drainage design
  from LiDAR (what shape the ground is, and where water will naturally flow
  or pool).

Together they sketch an end-to-end pipeline: **drone survey → digitised
features → terrain model → drainage plan**, all without an additional field
survey.

---

## Repository structure
IIT-Tirupati-Hackathon/
├── README.md
├── hack.ipynb                                    ← Problem Statement 1, FINAL pipeline
│                                                     (3-model segmentation ensemble)
├── Hackathon.ipynb                               ← Problem Statement 1, EARLIER pipeline
│                                                     (U-Net + Faster R-CNN, real training logs
│                                                      + saved visualisations)
│
├── DTM_OptimizedDrainageGeneration-main/
│   └── DTM_OptimizedDrainageGeneration-main/
│       ├── DTM_Multi_Train.ipynb                 ← Problem Statement 2, Stage A:
│       │                                            point cloud → TerrainNet → DTM
│       └── DTM2OptimizedDrainageSystem.ipynb      ← Problem Statement 2, Stage B:
│                                                     DTM → drainage network + waterlogging
│
└── SVAMITVA_Data_DTM_MULTI-20260623T143209Z-3-001/
└── SVAMITVA_Data_DTM_MULTI/
├── DSM_FILES/                            ← raw Digital Surface Models (6 villages)
├── DTM_FILES/                            ← model-generated Digital Terrain Models
├── SVAMITVA_Drainage_Output/              ← per-village drainage PNG + GeoJSON
├── all_dtms_preview.png
├── best_terrainnet.pth                   ← trained TerrainNet weights
└── GeoIntel Drainage optimization Submission Doc.pdf   ← official submission writeup

> `hack.ipynb` and `Hackathon.ipynb` both solve **Problem Statement 1** —
> they are two different generations of the same pipeline, not two separate
> problem statements. See the breakdown below for what changed between them.

---

## Problem Statement 1 — AI-Based Feature Extraction from Drone Orthophotos

> *"Develop an AI/ML model to identify key features from SVAMITVA Scheme
> drone orthophotos."* — official scope

**What it does:** given a village orthophoto, produce a pixel-level
classification into 10 classes:

| ID | Class | Source shapefile |
|----|-------|-------------------|
| 0 | Background | — |
| 1 | RCC Roof | `Built_Up_Area_type.shp` |
| 2 | Tiled Roof | `Built_Up_Area_type.shp` |
| 3 | Tin Roof | `Built_Up_Area_type.shp` |
| 4 | Thatched Roof | `Built_Up_Area_type.shp` |
| 5 | Road | `Road.shp` |
| 6 | Waterbody | `Water_Body.shp` |
| 7 | Transformer | `Utility.shp` |
| 8 | Tank | `Utility.shp` |
| 9 | Well | `Utility.shp` |

### `Hackathon.ipynb` — earlier two-headed pipeline

Run on an HPC cluster (`/nfsshare/users/manjula/svamitva_project`). Splits
the problem in two, because roofs/roads/water are **regions** while
transformers/tanks/wells are tiny **point-like objects** that plain
segmentation tends to miss:

| Stage | What happens | Output |
|---|---|---|
| 1 — Mask generation | Orthophoto TIFs + 5 SVAMITVA shapefiles rasterised into per-pixel label masks | `data/training/masks_raster/*.tif` |
| 2 — Patch extraction | Village rasters (up to 235,000×119,000 px) tiled into 256×256 patches, skipping mostly-empty tiles | `data/training/patches/{images,masks}/` |
| 3 — Minority-class augmentation | Rare classes (utilities, thatched roofs) oversampled so class imbalance doesn't dominate | same patch dirs |
| 4 — Train DuSA U-Net | Dual-attention U-Net trained for pixel-wise roof/road/water segmentation | `outputs/best_model.pth` |
| 5 — Train Faster R-CNN | Detector trained specifically on utility bounding boxes, with hard-negative mining | `outputs/final/faster_rcnn_utilities.pth` |
| 6 — Inference | Both models run over held-out villages | `outputs/predictions/`, `outputs/rcnn_utilities/` |
| 7 — Merge/evaluate/visualise | Segmentation + detections merged into one feature layer, metrics + charts produced | `outputs/final_predictions/` |

**Real logged results** (saved in the notebook's own outputs) — Faster
R-CNN utility detector, 40 epochs, 4,275 train / 1,069 val patches:

| Epoch | Precision | Recall |
|-------|-----------|--------|
| 1 | 100.00% | 0.42% |
| 10 | 88.81% | 82.30% |
| 20 | 91.87% | 85.24% |
| 40 | 91.39% | 86.70% |

CLI (defined inside the notebook itself):
```bash
python svamitva_pipeline.py --stage all      # everything
python svamitva_pipeline.py --stage 1        # masks only
python svamitva_pipeline.py --stage 1,2,3    # data prep only
python svamitva_pipeline.py --stage 4,5      # training only
```

### `hack.ipynb` — final ensemble pipeline

The final version drops the separate R-CNN head and instead handles **all
10 classes — including the tiny utility points — as segmentation**, using
lower confidence thresholds and smaller minimum-pixel counts for point-like
classes to compensate for their small footprint. Three architectures are
trained independently and combined with a weighted soft-voting ensemble.

**Models:**

| Model | Backbone | Why it's in the ensemble |
|---|---|---|
| `DuSAUNetPP` | Custom U-Net++ with **Channel + Spatial Dual Attention** blocks and an **ASPP** bottleneck (atrous rates 1/6/12/18), plus deep-supervision auxiliary heads | Sharp building/road boundaries, cheap to train from scratch |
| `SegFormerB5` | HuggingFace `nvidia/segformer-b5-finetuned-ade-640-640`, fine-tuned to 10 classes | Strong global context, good for large uniform regions |
| `Mask2FormerSeg` | HuggingFace `facebook/mask2former-swin-base-coco-panoptic`, fine-tuned to 10 classes | Query-based masks, good at separating adjacent roof instances |

`EnsembleModel` fuses the three softmax probability maps with tunable
weights `(w_segformer, w_mask2former, w_unetpp)`, default `(0.45, 0.35, 0.20)`.

**Loss function:**
loss = 0.3·LabelSmoothingCE + 0.4·DiceLoss + 0.2·FocalLoss(γ=2) + 0.1·BoundaryLoss
Class weights computed from inverse class frequency so rare utility classes
aren't drowned out (background ≈ 65% of pixels vs. transformer/well ≈ 0.6-0.7%).

**Training:** AdamW, `OneCycleLR`, separate (lower) LR for the pretrained
encoder vs. decoder/head, mixed precision + EMA weight averaging, 256×256
patches, batch size 8. Two villages (`MURDANDA`, `28996_NADALA`) are always
held out for validation.

**Inference:** 8-way test-time augmentation (rotation × flips), Gaussian-
weighted sliding-window stitching for seamless large rasters, per-class
confidence thresholds + minimum connected-component size + optional shape
regularisation, morphological cleanup, and automatic prediction
visualisations (input / segmentation / confidence heatmap).

**Ensemble weight tuning:** grid search over `(w1, w2, w3)` in 0.05 steps on
held-out villages, picks the combination maximising mean IoU → saved to
`outputs/best_weights.json`.

CLI:
```bash
python svamitva_pipeline.py --phase data       # rasterise shapefiles + tile patches
python svamitva_pipeline.py --phase train      # train all 3 models
python svamitva_pipeline.py --phase evaluate   # confusion matrices, per-class IoU/F1, charts
python svamitva_pipeline.py --phase tune       # grid-search ensemble weights
python svamitva_pipeline.py --phase predict    # run ensemble inference + export GeoPackage/visuals
python svamitva_pipeline.py --phase all        # all of the above in order
```

> **Path note:** both notebooks hard-code absolute paths from the team's own
> HPC training environment (e.g. `/nfsshare/users/P126014014/...`). Edit the
> `CFG` / `HPC_BASE` / `ROOT_DIR` constants near the top before re-running
> against your own data.

---

## Problem Statement 2 — DTM Generation & Optimized Drainage Network

> *"Develop an AI/ML-driven workflow to generate DTMs from drone point-cloud
> data and design drainage networks for densely inhabited village (abadi)
> areas."* — official scope

**Team:** Harshini S, Gayathri Gopal Peshwa, Kaaviya Varshini T (SASTRA
Deemed University), advised by Dr. S. Kamakshi and Dr. K.R. Manjula.

### Stage A — `DTM_Multi_Train.ipynb`: Point Cloud → DTM

1. **Load point cloud** (`laspy`) — up to ~9-23 million raw points per
   village. A stratified spatial-bin sampler caps this at **400,000 points**
   while keeping even spatial coverage.
2. **Ground classification** — the **Cloth Simulation Filter (CSF)** drapes
   a virtual cloth over the inverted point cloud; points it settles on are
   classified as ground vs. non-ground (buildings/vegetation). Example run:
   9.8M raw points → 396K sampled → 65.5% classified as ground.
3. **Terrain roughness weighting** — a reference DTM is smoothed (5×5
   uniform filter) and compared to the raw DTM to get a per-pixel roughness
   map; points in rough/noisy cells get lower training weight
   (`reliable=1.0`, `moderate=0.5`, `noisy=0.1`).
4. **Feature extraction** — 20-D feature vector per point from raw/centered/
   normalised XYZ plus **20-nearest-neighbour** statistics (`KDTree`):
   mean/std/min neighbour height, height above local mean/min, CSF ground
   label, intensity, colour, return number.
5. **Blocking** — points grouped into 10 m × 10 m spatial blocks with 50%
   overlap (min. 64 points/block); targets z-score normalised per block.
6. **`TerrainNet` model** — PointNet-style: a **point branch** (1×1 Conv1D)
   encoding each point individually, a **global branch** pooling
   (max+mean+std) for block-level context, and a **neighbour branch** (1×1
   Conv2D over each point's k-NN neighbourhood) for local geometry — all
   three concatenated and regressed down to a predicted elevation per point.
7. **Loss** — Huber loss (δ=0.5) weighted by `CSF ground-label × terrain
   roughness`, plus a gradient-consistency term penalising inconsistent
   elevation *differences* between neighbours (keeps terrain smooth).
8. **Optimisation** — two-phase schedule: **AdamW + OneCycleLR** for the
   first 70% of epochs, then **SGD + Nesterov momentum + Cosine Annealing**
   for the remaining 30%.
9. **Augmentation** — random Z-axis rotation, small Z jitter, random point
   dropout per batch.
10. **Multi-village generalisation** — trained jointly on 6 villages across
    **Rajasthan, Punjab and Gujarat**; **KHAPRETA (Gujarat)** held out
    completely.

**Result:** MAE ≈ **0.117 m** on the fully held-out KHAPRETA village —
i.e. predicted ground elevation is, on average, within ~11.7 cm of the
reference DTM on a village/state never seen during training.

### Stage B — `DTM2OptimizedDrainageSystem.ipynb`: DTM → Drainage Network

Runs classic D8 hydrology using **`pysheds`** on the DTMs from Stage A:

1. **Hydrological conditioning** — `fill_pits` → `fill_depressions` →
   `resolve_flats`, removing artifacts that would trap simulated flow.
2. **Flow direction & accumulation** — D8 flow direction, then flow
   accumulation per cell.
3. **Multi-tier drainage classification** by flow-accumulation percentile:
   **Primary** (top ~1-2%), **Secondary** (next ~2-3%), **Tertiary** (next
   ~5-8%) — two percentile presets included (98/95/90 and 99/97/95).
4. **Waterlogging hotspot detection** — flagged where **low slope** (bottom
   25th pct) + **high flow accumulation** (top 75th pct) + **low elevation**
   (bottom 30th pct) coincide.
5. **Vectorisation & export** — each class exported as point geometries in
   a single combined **GeoJSON** per village, directly usable in
   QGIS/ArcGIS/Google Earth Engine.
6. **Visualisation** — PNG overlay per village: normalised DTM in greyscale
   with primary (cyan), secondary (orange), tertiary (green) and
   waterlogging (red) layers on top.

Batch mode loops `process_dtm()` over every `.tif` in the input folder.

---

## Getting started

### 1. Clone and set up an environment

```bash
git clone <this-repo-url>
cd IIT-Tirupati-Hackathon
python3 -m venv venv && source venv/bin/activate    # or use conda
```

### 2. Install dependencies

```bash
# Problem Statement 1 — hack.ipynb / Hackathon.ipynb
pip install torch torchvision transformers albumentations opencv-python \
            rasterio geopandas shapely scikit-learn seaborn matplotlib pillow numpy

# Problem Statement 2 — DTM_OptimizedDrainageGeneration-main/
pip install laspy lazrs cloth-simulation-filter torch scikit-learn scipy \
            rasterio pysheds geopandas shapely scikit-image matplotlib pandas numpy
```

`rasterio` / `geopandas` / `pysheds` all depend on **GDAL** being installed
at the OS level first (`apt install gdal-bin libgdal-dev` on Ubuntu,
`brew install gdal` on macOS).

### 3. Run

Open `hack.ipynb`, `Hackathon.ipynb`, `DTM_Multi_Train.ipynb` or
`DTM2OptimizedDrainageSystem.ipynb` in Jupyter/Colab and run top-to-bottom.

> Both problem statements' code has **absolute paths hard-coded** near the
> top (HPC cluster paths for PS1, Google Drive paths for PS2, since
> `DTM_Multi_Train.ipynb` and `DTM2OptimizedDrainageSystem.ipynb` were
> developed on Google Colab with `google.colab.drive` mounted). Update the
> path constants (`CFG`, `HPC_BASE`, `ROOT_DIR`, `LAZ_FILES`, `DTM_FILES`,
> `INPUT_DIR`, `OUTPUT_DIR`, etc.) before running against your own data.

---

## Data

- **Problem Statement 1** input (orthophoto GeoTIFFs + shapefiles) is not
  bundled in this repo — it's multi-GB drone imagery distributed by MoPR/NIC
  for the hackathon.
- **Problem Statement 2** input/output is partly bundled in
  `SVAMITVA_Data_DTM_MULTI-20260623T143209Z-3-001/`, covering six villages —
  **67169_5NKR_CHAKHIRASINGH** (Rajasthan), **Dhal_Hoshiarpur** (Punjab),
  **KHAPRETA** (Gujarat, held-out test), **DHUNDA_FATEHGARH SAHIB**
  (Punjab), **64334_2H (REFLIGHT)** (Rajasthan), **DEVDI** (Gujarat):
  - `DSM_FILES/` — raw Digital Surface Models
  - `DTM_FILES/` — model-generated Digital Terrain Models
  - `SVAMITVA_Drainage_Output/` — per-village drainage PNG + GeoJSON layers
  - `best_terrainnet.pth` — trained TerrainNet weights
  - `all_dtms_preview.png` — preview of all six generated DTMs
  - `GeoIntel Drainage optimization Submission Doc.pdf` — the official
    hackathon submission writeup, including the full 5-stage methodology,
    tech stack, results and roadmap (this README summarises it above)

For the full source dataset (raw point clouds, additional villages), see the
Drive link referenced in the submission PDF.

---

## Results summary

| Metric | Value | Where it comes from |
|---|---|---|
| Faster R-CNN utility detector — precision | **91.4%** (epoch 40) | `Hackathon.ipynb`, real training log |
| Faster R-CNN utility detector — recall | **86.7%** (epoch 40) | `Hackathon.ipynb`, real training log |
| TerrainNet ground-elevation MAE on held-out village | **≈ 0.117 m** | KHAPRETA (Gujarat), never seen during training — official submission document |
| CPU-only DTM + drainage inference time | **< 10 minutes / village** | Official submission document |

---

## Tech stack

| Layer | Tools |
|---|---|
| Deep learning | PyTorch, HuggingFace Transformers (SegFormer, Mask2Former) |
| Geospatial I/O & analysis | rasterio, GDAL, geopandas, shapely, pysheds |
| Point cloud processing | laspy / lazrs, Cloth Simulation Filter (CSF) |
| Classical ML / features | scikit-learn (KDTree, metrics), scipy |
| Image processing / augmentation | OpenCV, Albumentations, PIL |
| Visualisation | Matplotlib, Seaborn |
| Compute environments used | HPC cluster (Problem Statement 1 training), Google Colab (Problem Statement 2 training) |

Everything is built on **open-source libraries only** — no proprietary GIS
licenses required, which matters for a solution meant to eventually run at
the scale of India's 600,000+ villages.

---
