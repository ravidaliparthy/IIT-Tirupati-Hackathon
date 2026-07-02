# IIT-Tirupati-Hackathon

**AI/ML Hackathon — Ministry of Panchayati Raj (MoPR), Government of India**
Organised by the **Geospatial Intelligence & Applications Laboratory, IIT
Tirupati Navavishkar I-Hub Foundation**, in partnership with **IIIT
Tirupati** and the **National Informatics Centre (NIC)**.

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

- [Background — what is SVAMITVA and why does this matter](#background--what-is-svamitva-and-why-does-this-matter)
- [The problem, in plain terms](#the-problem-in-plain-terms)
- [Repository structure](#repository-structure)
- [Problem Statement 1 — Feature Extraction](#problem-statement-1--ai-based-feature-extraction-from-drone-orthophotos)
- [Problem Statement 2 — DTM & Drainage](#problem-statement-2--dtm-generation--optimized-drainage-network)
- [Glossary — terms used throughout this repo](#glossary--terms-used-throughout-this-repo)
- [Getting started](#getting-started)
- [Data](#data)
- [Results summary](#results-summary)
- [Tech stack](#tech-stack)
- [Acknowledgements](#acknowledgements)

---

## Background — what is SVAMITVA and why does this matter

**SVAMITVA** (Survey of Villages Abadi and Mapping with Improvised
Technology in Village Areas) is a Government of India scheme that uses
drones to survey the *abadi* (inhabited) areas of rural villages — something
that had never been comprehensively mapped before. Each drone flight
produces two kinds of raw data:

- A high-resolution **orthophoto** — essentially a distortion-corrected
  aerial photograph, accurate enough that a building's edges on the image
  line up with its true position on the ground.
- For some villages, a **LiDAR point cloud** — millions of individual 3D
  points (x, y, z + extra attributes like intensity and return number)
  captured by a laser scanner mounted on the drone, dense enough to
  reconstruct both the ground surface and everything sitting on it
  (buildings, trees, poles).

This raw data is extremely rich, but on its own it's just pixels and point
coordinates — it doesn't tell a Panchayat official *which* buildings exist,
*what* they're made of, *where* the roads and water bodies are, or *how*
water will flow across the terrain during monsoon. Turning raw drone data
into that kind of actionable planning information has historically required
GIS technicians to manually trace features on screen, village by village —
a process that does not scale to the 600,000+ villages SVAMITVA eventually
aims to cover.

This hackathon challenged teams to close that gap with AI/ML, working with
real SVAMITVA data from multiple Indian states.

---

## The problem, in plain terms

Two related but distinct problems were posed, and this repository answers
both:

**Problem Statement 1** asks: *"Given a drone photo of a village, can a
model automatically tell us what's on the ground?"* — i.e., look at the
orthophoto the way a human GIS technician would, and label every pixel as
one of: background, a specific roof material (RCC/tiled/tin/thatched), road,
waterbody, or one of three small utility structures (transformer, overhead
tank, well). This is a **computer vision / semantic segmentation** problem.

**Problem Statement 2** asks: *"Given a raw LiDAR scan of a village, can a
model figure out the true shape of the bare ground (ignoring buildings and
trees), and then tell us where rainwater will naturally flow or pool?"*
This has two parts: first a **3D point-cloud regression** problem (predict
ground elevation at every location), and then a **classical hydrology
simulation** on top of that predicted terrain to trace drainage paths and
flag flood/waterlogging risk zones.

The two problems are complementary pieces of the same larger picture: once
you know *what* is on the land (Problem Statement 1) and *what shape* the
land is (Problem Statement 2), a Panchayat has enough information to plan
infrastructure — including drainage systems that actually work with the
terrain instead of against it — without commissioning a brand-new survey.

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
> problem statements. The breakdown below explains what changed between
> them and why.

---

## Problem Statement 1 — AI-Based Feature Extraction from Drone Orthophotos

> *"Develop an AI/ML model to identify key features from SVAMITVA Scheme
> drone orthophotos."* — official scope

### What "feature extraction" means here

Every pixel in a village orthophoto is assigned exactly one of 10 labels.
This is a **10-class semantic segmentation** problem — the model doesn't
just say "there's a building somewhere in this image", it draws the precise
outline of every building, road and water body, and additionally decides
*what kind* of roof each building has (which matters a lot for rural
planning, since RCC/tiled/tin/thatched roofs imply very different levels of
housing quality and flood/fire vulnerability).

| ID | Class | Source shapefile | Why it matters for planning |
|----|-------|-------------------|------------------------------|
| 0 | Background | — | Everything that isn't a mapped feature |
| 1 | RCC Roof | `Built_Up_Area_type.shp` | Pucca (permanent, concrete) housing |
| 2 | Tiled Roof | `Built_Up_Area_type.shp` | Semi-permanent housing |
| 3 | Tin Roof | `Built_Up_Area_type.shp` | Semi-permanent housing, heat/fire risk |
| 4 | Thatched Roof | `Built_Up_Area_type.shp` | Kutcha (temporary) housing, highest vulnerability |
| 5 | Road | `Road.shp` | Access & connectivity planning |
| 6 | Waterbody | `Water_Body.shp` | Water resource & flood-risk mapping |
| 7 | Transformer | `Utility.shp` | Electrical infrastructure inventory |
| 8 | Tank | `Utility.shp` | Water storage infrastructure inventory |
| 9 | Well | `Utility.shp` | Water source infrastructure inventory |

The ground-truth labels for training aren't hand-drawn from scratch — they
come from **rasterising the SVAMITVA cadastral shapefiles** (vector polygons
already digitised for the surveyed villages) directly onto the orthophoto's
pixel grid, so every training pixel inherits the label of whatever
shapefile polygon it falls inside.

### `Hackathon.ipynb` — the earlier, two-headed pipeline

This was the first working end-to-end version, developed and run on an HPC
cluster (`/nfsshare/users/manjula/svamitva_project`). Its key design insight
is that the 10 classes aren't all the same *shape* of problem:

- Roofs, roads and waterbodies are **large, contiguous regions** — a
  standard segmentation network handles these well.
- Transformers, tanks and wells are **tiny, sparse, point-like objects**
  (sometimes just a few pixels wide) — a segmentation network trained mostly
  on huge background/road/roof regions tends to simply ignore them, because
  getting them wrong barely moves the loss.

So this version **splits the problem into two specialised models** trained
independently:

| Stage | What happens | Output |
|---|---|---|
| 1 — Mask generation | Orthophoto TIFs + the 5 SVAMITVA shapefiles are rasterised into per-pixel label masks, one mask per village | `data/training/masks_raster/*.tif` |
| 2 — Patch extraction | Village rasters — some as large as 235,000 × 119,000 pixels, far too big to feed a network directly — are tiled into 256×256 patches, skipping tiles that are almost entirely empty background | `data/training/patches/{images,masks}/` |
| 3 — Minority-class augmentation | Rare classes (utilities, thatched roofs) are deliberately oversampled/augmented so they aren't drowned out by the overwhelmingly common background/roof pixels | same patch dirs |
| 4 — Train DuSA U-Net | A dual-attention U-Net is trained purely for pixel-wise roof/road/water segmentation | `outputs/best_model.pth` |
| 5 — Train Faster R-CNN | A **Faster R-CNN object detector** — a completely different architecture, designed for finding bounding boxes rather than pixel masks — is trained specifically to localise transformers/tanks/wells, using hard-negative mining (deliberately showing it background patches that superficially resemble utilities) to cut down false positives | `outputs/final/faster_rcnn_utilities.pth` |
| 6 — Inference | Both trained models are run independently over held-out test villages | `outputs/predictions/`, `outputs/rcnn_utilities/` |
| 7 — Merge / evaluate / visualise | The U-Net's region segmentation and the R-CNN's utility detections are merged into a single combined feature layer, and evaluation metrics + comparison charts are generated | `outputs/final_predictions/` |

**Real logged training results** (captured directly in the notebook's saved
outputs, not simulated) — Faster R-CNN utility detector, 40 epochs over
4,275 training / 1,069 validation patches:

| Epoch | Precision | Recall | What this tells us |
|-------|-----------|--------|----------------------|
| 1 | 100.00% | 0.42% | Model is barely detecting anything yet — every prediction it *does* make happens to be correct, but it's finding almost none of the true utilities |
| 10 | 88.81% | 82.30% | Rapid early improvement as the detector learns what a transformer/tank/well looks like |
| 20 | 91.87% | 85.24% | Gains are slowing — the model is converging |
| 40 | 91.39% | 86.70% | Final plateau: roughly 9 in 10 predicted utility detections are correct, and the model finds about 87% of all true utilities in the validation set |

This precision/recall trade-off curve — starting at perfect-but-useless
precision and gradually trading a little precision for much more recall — is
a classic signature of a detector learning to loosen an overly conservative
initial decision boundary.

CLI (defined inside the notebook itself):
```bash
python svamitva_pipeline.py --stage all      # everything, stages 1 through 7
python svamitva_pipeline.py --stage 1        # masks only
python svamitva_pipeline.py --stage 1,2,3    # data preparation only
python svamitva_pipeline.py --stage 4,5      # training only
```

### `hack.ipynb` — the final, unified ensemble pipeline

The final version rethinks the two-headed approach. Rather than maintaining
two entirely different architectures (and two entirely different inference
paths that then need to be merged), it treats **all 10 classes — including
the tiny utility points — as one segmentation problem**, and compensates for
the size imbalance a different way: by giving each class its own
confidence threshold and minimum-detection-size at inference time (a
transformer only needs a handful of confident pixels to count as "found";
a road needs a much larger contiguous blob before it's trusted). Three
different segmentation architectures are trained completely independently
and then combined with a weighted vote — the reasoning being that different
architectures make *different kinds* of mistakes, so combining them
smooths out each one's individual weaknesses.

**The three models in the ensemble, and what each contributes:**

| Model | Backbone | Architectural idea | Why it's in the ensemble |
|---|---|---|---|
| `DuSAUNetPP` | Custom U-Net++ style encoder-decoder built from scratch, with **Channel Attention** (learns *which feature channels* matter most) and **Spatial Attention** (learns *which spatial locations* matter most) blocks at every decoder stage, plus an **ASPP** bottleneck (Atrous Spatial Pyramid Pooling — convolutions at multiple dilation rates 1/6/12/18 to see the image at multiple zoom levels simultaneously) and auxiliary supervision heads partway through the decoder | Trained entirely from scratch on this data, so it isn't biased by whatever a general-purpose pretrained model was originally trained to look for | Produces the sharpest object *boundaries* — important for accurately measuring building footprint area |
| `SegFormerB5` | HuggingFace `nvidia/segformer-b5-finetuned-ade-640-640`, a transformer-based segmentation model, fine-tuned from its pretrained weights down to 10 classes | Self-attention lets every pixel "see" the entire image, not just a local neighbourhood | Best at understanding large-scale context — e.g. correctly extending a waterbody's boundary even where it's partly obscured |
| `Mask2FormerSeg` | HuggingFace `facebook/mask2former-swin-base-coco-panoptic`, a query-based segmentation model, fine-tuned to 10 classes | Instead of classifying pixels directly, it learns a fixed set of "queries" that each learn to find one object instance and its mask | Best at separating buildings that sit right next to each other, which pure per-pixel classifiers often blur together |

The `EnsembleModel` class combines the three models' softmax probability
maps with tunable weights `(w_segformer, w_mask2former, w_unetpp)`,
defaulting to `(0.45, 0.35, 0.20)` — SegFormer's global context is trusted
most, followed by Mask2Former, with the from-scratch U-Net contributing the
least (but still meaningfully, especially for boundary precision).

**Loss function** — training uses a `CombinedLoss` that blends four
different objectives, because no single loss captures everything a good
segmentation model needs:
loss = 0.3 · LabelSmoothingCE   (standard classification loss, softened
slightly so the model isn't overconfident)
+ 0.4 · DiceLoss           (directly rewards overlap between predicted
and true regions — robust to class imbalance)
+ 0.2 · FocalLoss(γ=2)     (down-weights easy/already-correct pixels so
training focuses on hard, ambiguous ones)
+ 0.1 · BoundaryLoss       (specifically penalises errors right at
object edges, where segmentation models
usually struggle most)
Per-class weights inside these losses are derived from inverse class
frequency, since background pixels dominate the dataset (≈65% of all
pixels) while a class like "well" makes up under 1% — without this
weighting, a model could reach high overall accuracy just by predicting
background everywhere and never learning the rare classes at all.

**Training setup:** AdamW optimiser with a `OneCycleLR` schedule (learning
rate ramps up then anneals down over training, which tends to reach better
minima faster than a flat rate); the pretrained encoder gets a lower
learning rate than the newly-initialised decoder/head, since the encoder
already "knows" general visual features and shouldn't be disturbed too
aggressively. Training runs in mixed precision (`torch.cuda.amp`) for
speed, and an EMA (Exponential Moving Average) shadow copy of the weights is
maintained and used for evaluation — EMA weights tend to generalise better
than the raw end-of-training weights because they average out the last few
epochs' worth of noisy updates. Patches are 256×256, batch size 8, and two
entire villages (`MURDANDA`, `28996_NADALA`) are permanently excluded from
training and reserved purely for validation, so reported metrics reflect
performance on villages the model has genuinely never seen — a much more
honest test than validating on held-out patches from villages it *has*
partially seen.

**Inference-time techniques** — several tricks are layered on top of the
raw model predictions to squeeze out extra accuracy without any retraining:

- **Test-Time Augmentation (TTA):** the same image is fed through the model
  8 times (combinations of horizontal flip, vertical flip, and 90° rotation)
  and the predictions are averaged. Since the "true" segmentation shouldn't
  change if you flip or rotate the input, averaging these 8 predictions
  cancels out orientation-specific errors.
- **Gaussian-weighted sliding-window stitching:** because full village
  orthophotos are far too large to process in one shot, they're processed
  patch-by-patch with overlapping windows, and each patch's contribution to
  the final map is weighted by a Gaussian centred on the patch (full
  confidence in the middle, tapering off towards the edges) — this avoids
  the visible "seams" you'd otherwise see at patch boundaries.
- **Per-class post-processing (`CLASS_CFG`):** each of the 10 classes gets
  its own confidence threshold and minimum connected-component pixel count
  before a detection is kept, plus an optional shape-regularisation step for
  roof classes (nudging blobby predictions towards cleaner polygon shapes) —
  because a correct detection threshold for a road (which might be one
  contiguous strip hundreds of pixels wide) would completely miss a
  transformer (which might only be 3-4 pixels across).
- **Morphological cleanup:** small stray misclassified pixels are removed
  with standard open/close morphological operations, and every prediction
  is automatically visualised as a three-panel image (raw input orthophoto
  / predicted segmentation / max-confidence heatmap) for quick visual QA.

**Ensemble weight tuning:** after all three models are trained
independently, a dedicated tuning step does a grid search over ensemble
weights `(w1, w2, w3)` in steps of 0.05 across the held-out validation
villages, and keeps whichever combination maximises mean IoU (Intersection
over Union — see the glossary below) — this is more reliable than simply
guessing weights, since the "best" model isn't necessarily the one that
should get the most trust in every scenario.

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
> HPC training environment (e.g. `/nfsshare/users/P126014014/...`). Before
> re-running against your own data, edit the `CFG` / `HPC_BASE` / `ROOT_DIR`
> constants near the top of the relevant notebook.

---

## Problem Statement 2 — DTM Generation & Optimized Drainage Network

> *"Develop an AI/ML-driven workflow to generate DTMs from drone point-cloud
> data and design drainage networks for densely inhabited village (abadi)
> areas."* — official scope

### What this problem is actually solving

A raw LiDAR scan captures the height of *everything* the laser hits — roofs,
treetops, power lines, as well as the actual ground. If you want to
understand how water will flow across a village during heavy rain, none of
that "clutter" is useful — you specifically need the **bare-earth
elevation**, stripped of every building and every tree. Reconstructing that
bare-earth surface from a cluttered point cloud is Stage A of this pipeline.
Once you have that clean terrain surface, Stage B applies well-established
hydrology math to it — the same category of algorithm used in professional
watershed-analysis GIS software — to trace where rainwater would naturally
converge into channels, and where it would pool because it has nowhere to
drain.

### Stage A — `DTM_Multi_Train.ipynb`: Point Cloud → DTM

1. **Load the point cloud** (`laspy`). Raw village scans in this dataset
   range from about 9.8 million to 23.4 million points. Since that's far too
   many to train on directly, a **stratified spatial-bin sampler** reduces
   each village down to a manageable 400,000 points while deliberately
   preserving even coverage across the whole village footprint — a naive
   random sample would otherwise over-represent densely-scanned areas and
   under-represent sparse ones.
2. **Ground classification with CSF.** The **Cloth Simulation Filter**
   imagines flipping the point cloud upside down and draping a virtual
   piece of cloth over it under gravity — the cloth settles onto the
   "highest" points from this inverted view, which correspond to the ground
   surface in the original orientation (since gravity would otherwise pull
   the cloth down onto the true ground, but buildings and trees block it
   from reaching there). Points the cloth rests on are labelled ground;
   everything else (buildings, vegetation) is labelled non-ground. In one
   example village run, 9.8M raw points sampled down to 396K resulted in
   65.5% being classified as ground.
3. **Terrain roughness weighting.** A reference DTM is smoothed with a 5×5
   uniform filter, and the difference between the smoothed and raw surface
   gives a per-pixel "roughness" score. Areas with high roughness usually
   indicate noisy, less-trustworthy elevation data (rather than genuinely
   jagged terrain), so points sampled there are down-weighted during
   training: `reliable → weight 1.0`, `moderate → weight 0.5`,
   `noisy → weight 0.1`. This stops the model from being misled by bad
   ground-truth data instead of learning genuine terrain patterns.
4. **Feature extraction.** Every point gets a 20-dimensional feature vector:
   its raw XYZ position, a centered version, a normalised version, plus
   statistics computed from its **20 nearest spatial neighbours** (via
   `KDTree`) — mean/std/min neighbour height, and how far above the local
   mean/minimum the point sits — along with its CSF ground label, laser
   return intensity, colour, and return number where available. These
   neighbourhood statistics are what let the model reason about local
   terrain shape rather than just memorising isolated elevation values.
5. **Blocking.** Points are grouped into overlapping 10 m × 10 m spatial
   blocks (50% overlap, minimum 64 points per block), and target elevations
   within each block are z-score normalised — this lets the network learn
   *relative* terrain shape within a local neighbourhood rather than
   absolute elevation values, which vary hugely from village to village.
6. **The `TerrainNet` model** — a PointNet-style neural network purpose
   built for unordered 3D point data (unlike images, point clouds have no
   fixed grid, so ordinary convolutions don't directly apply). It combines
   three branches: a **point branch** that encodes each point independently
   through 1×1 convolutions, a **global branch** that pools the point
   branch's output (via max, mean and standard deviation) to summarise the
   whole block's context, and a **neighbour branch** that processes each
   point's local k-nearest-neighbour patch through 2D convolutions to
   capture fine local geometry. All three branches' outputs are
   concatenated and passed through fully-connected layers down to a single
   predicted elevation value per point.
7. **Loss function.** A Huber loss (robust to outliers, unlike plain MSE)
   weighted by the combination of CSF ground-label confidence and terrain
   roughness reliability, plus a gradient-consistency term that penalises
   the model when the *differences* in elevation between neighbouring
   predicted points don't match the *differences* in the true elevations —
   this keeps the predicted terrain surface physically smooth rather than
   noisy point-to-point.
8. **Two-phase optimisation schedule.** Training starts with **AdamW +
   OneCycleLR** for the first 70% of epochs (fast, adaptive early
   convergence), then switches to **SGD with Nesterov momentum + Cosine
   Annealing** for the remaining 30% — a well-known trick where SGD's
   simpler, noisier updates in the late stages tend to find flatter, better-
   generalising minima than AdamW alone.
9. **Data augmentation.** Each training batch is randomly rotated around
   the vertical (Z) axis, given a small amount of elevation jitter, and has
   a fraction of its points randomly dropped and resampled — all of which
   make the model robust to the arbitrary orientation and point density it
   will encounter on new, unseen villages.
10. **Multi-village training with a genuine held-out test.** The model is
    trained jointly across six villages spanning **three different Indian
    states — Rajasthan, Punjab and Gujarat** — so it has to learn terrain
    patterns general enough to transfer across regions, not just memorise
    one local landscape. **KHAPRETA (Gujarat) is excluded entirely from
    training** and used purely as a final test of generalisation.

**Result:** on KHAPRETA, the model achieves a **Mean Absolute Error of
approximately 0.117 metres** — meaning its predicted ground elevation is, on
average, within about 11.7 centimetres of the true reference DTM, on a
village and terrain type it never saw during training. For context, this is
well within the accuracy needed for village-scale drainage planning.

### Stage B — `DTM2OptimizedDrainageSystem.ipynb`: DTM → Drainage Network

This stage takes the DTM produced by Stage A (or any bare-earth GeoTIFF) and
runs it through standard hydrological flow-modelling using **`pysheds`** —
the same underlying algorithm family used in professional watershed
analysis tools:

1. **Hydrological conditioning** — `fill_pits` → `fill_depressions` →
   `resolve_flats`. Real-world (and predicted) elevation data almost always
   contains small artificial pits and flat spots that would otherwise trap
   simulated water in a single pixel forever; this conditioning step removes
   those artifacts so flow can be traced continuously across the whole
   surface.
2. **Flow direction and accumulation.** Using the **D8 method** (each cell's
   water is assumed to flow into whichever one of its 8 neighbouring cells
   is steepest downhill), the algorithm computes a flow-direction grid, then
   a flow-accumulation grid — the number of upstream cells that eventually
   drain through each cell. High accumulation values mark natural channels;
   low values mark ridges and slopes.
3. **Multi-tier drainage classification.** Flow-accumulation values are
   split into percentile-based tiers: **Primary channels** (roughly the top
   1-2% of accumulation — the main drainage lines that carry the most
   water), **Secondary channels** (the next 2-3%), and **Tertiary channels**
   (the next 5-8%, smaller feeder channels). Two percentile presets are
   included in the notebook (a stricter 98/95/90 split and a looser
   99/97/95 split) so the density of the resulting drainage network can be
   tuned to how conservative or comprehensive the output should be.
4. **Waterlogging hotspot detection.** A location is flagged as a
   waterlogging risk when three conditions coincide: **low slope** (bottom
   25th percentile — the ground is nearly flat), **high flow accumulation**
   (top 75th percentile — a lot of water naturally converges there), and
   **low elevation** (bottom 30th percentile — it sits in a local low
   point). Flat, low-lying spots that water flows into but can't easily
   flow out of are exactly where waterlogging and localised flooding tend
   to occur in practice.
5. **Vectorisation and export.** Every classified cell (primary/secondary/
   tertiary/waterlog) is converted from a raster pixel into a point
   geometry and exported as a single combined **GeoJSON** file per village
   — a standard vector format that opens directly in QGIS, ArcGIS, or
   Google Earth Engine without any conversion step.
6. **Visualisation.** A PNG is generated per village showing the normalised
   DTM in greyscale as a base layer, with the primary (cyan), secondary
   (orange), tertiary (green) drainage channels and waterlogging zones (red)
   overlaid — dilated slightly so thin channels remain visible at a glance.

The whole batch process simply loops `process_dtm()` over every `.tif` file
found in the input folder, so an entire state's worth of processed villages
can be run through in one unattended pass.

---

## Glossary — terms used throughout this repo

For anyone newer to geospatial ML, a quick reference for terms used above:

| Term | Meaning |
|---|---|
| **Orthophoto** | An aerial photo that's been geometrically corrected so every pixel's position accurately matches its true location on the ground (unlike a raw, uncorrected photo, which distorts near the edges) |
| **DSM (Digital Surface Model)** | An elevation model of the *top* surface — includes roofs, trees, everything the sensor first hits |
| **DTM (Digital Terrain Model)** | An elevation model of the *bare ground only*, with buildings and vegetation removed — what Stage A of Problem Statement 2 predicts |
| **Point cloud** | A set of individual 3D points (x, y, z, plus extra attributes) captured by a laser scanner (LiDAR); has no fixed grid structure, unlike an image |
| **Semantic segmentation** | A computer vision task where every pixel of an image is classified into a category, producing a full labelled map rather than just a single label for the whole image |
| **Shapefile (`.shp`)** | A common GIS vector file format storing geographic features as points/lines/polygons with attached attribute data |
| **Rasterise** | Convert vector data (like shapefile polygons) into a pixel grid, so it can be compared against or overlaid on an image |
| **IoU (Intersection over Union)** | A segmentation accuracy metric: the overlap area between predicted and true regions, divided by their combined area — 100% means a perfect match |
| **Mean IoU (mIoU)** | The IoU averaged across all classes, commonly used as the single headline number for comparing segmentation models |
| **Precision** | Of everything the model predicted as positive, what fraction was actually correct |
| **Recall** | Of everything that was actually positive, what fraction the model successfully found |
| **CSF (Cloth Simulation Filter)** | An algorithm that separates ground points from non-ground points in a LiDAR point cloud by simulating a cloth draping over the (inverted) terrain |
| **D8 flow direction** | A hydrology algorithm that assigns each terrain cell's water flow to whichever one of its 8 neighbouring cells is steepest downhill |
| **Flow accumulation** | For each cell in a terrain grid, the number of upstream cells whose water eventually flows through it — high values indicate natural drainage channels |
| **Ensemble (model)** | Combining the predictions of multiple independently-trained models, usually by averaging or voting, to get a result that's more robust than any single model alone |
| **Test-Time Augmentation (TTA)** | Running the same input through a model multiple times with small transformations (flips/rotations) applied, then averaging the results, to reduce orientation-specific prediction errors |
| **EMA (Exponential Moving Average)** | Keeping a running, smoothed average of a model's weights throughout training, which often generalises better than the raw final-epoch weights |

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
`brew install gdal` on macOS) — installing the Python packages alone isn't
enough.

### 3. Run

Open `hack.ipynb`, `Hackathon.ipynb`, `DTM_Multi_Train.ipynb` or
`DTM2OptimizedDrainageSystem.ipynb` in Jupyter or Google Colab and run
top-to-bottom.

> Both problem statements' code has **absolute paths hard-coded** near the
> top — HPC cluster paths for Problem Statement 1, and Google Drive paths
> for Problem Statement 2 (since `DTM_Multi_Train.ipynb` and
> `DTM2OptimizedDrainageSystem.ipynb` were developed on Google Colab with
> `google.colab.drive` mounted). Before running against your own data,
> update the relevant path constants (`CFG`, `HPC_BASE`, `ROOT_DIR`,
> `LAZ_FILES`, `DTM_FILES`, `INPUT_DIR`, `OUTPUT_DIR`, etc.) near the top of
> each notebook.

---

## Data

- **Problem Statement 1** input (orthophoto GeoTIFFs + shapefiles) is not
  bundled in this repo — it's multi-GB drone imagery distributed by MoPR/NIC
  specifically for the hackathon.
- **Problem Statement 2** input/output is partly bundled in
  `SVAMITVA_Data_DTM_MULTI-20260623T143209Z-3-001/`, covering six villages —
  **67169_5NKR_CHAKHIRASINGH** (Rajasthan), **Dhal_Hoshiarpur** (Punjab),
  **KHAPRETA** (Gujarat, held-out test village), **DHUNDA_FATEHGARH SAHIB**
  (Punjab), **64334_2H (REFLIGHT)** (Rajasthan), and **DEVDI** (Gujarat):
  - `DSM_FILES/` — raw Digital Surface Models
  - `DTM_FILES/` — model-generated Digital Terrain Models
  - `SVAMITVA_Drainage_Output/` — per-village drainage PNG + GeoJSON layers
  - `best_terrainnet.pth` — trained TerrainNet weights, ready to load directly
  - `all_dtms_preview.png` — a single preview image of all six generated DTMs side by side
  - `GeoIntel Drainage optimization Submission Doc.pdf` — the official
    hackathon submission writeup, covering the full 5-stage methodology,
    technology stack, results, and future roadmap (much of this README's
    Problem Statement 2 section is drawn directly from it)

For the full source dataset — including additional villages and the raw
point clouds — see the Google Drive link referenced inside the submission
PDF.

---

## Results summary

| Metric | Value | Where it comes from |
|---|---|---|
| Faster R-CNN utility detector — precision | **91.4%** (epoch 40) | `Hackathon.ipynb`, real training log |
| Faster R-CNN utility detector — recall | **86.7%** (epoch 40) | `Hackathon.ipynb`, real training log |
| TerrainNet ground-elevation MAE on held-out village | **≈ 0.117 m** | KHAPRETA (Gujarat), never seen during training — official submission document |
| CPU-only DTM + drainage inference time | **< 10 minutes / village** | Official submission document |

The final ensemble segmentation pipeline (`hack.ipynb`) additionally
generates a full evaluation suite whenever `--phase evaluate` or
`--phase tune` is run — per-class IoU/F1/precision/recall bar charts,
normalised confusion matrices, and side-by-side model comparison plots —
useful for understanding exactly which classes each of the three ensemble
members is strongest and weakest at.

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

Every dependency here is **open-source**, with no proprietary GIS licenses
required — a deliberate choice, since a solution meant to eventually run
across India's 600,000+ villages needs to be economically viable to deploy
at that scale.

---

## Acknowledgements

- **Ministry of Panchayati Raj (MoPR)**, Government of India — hackathon
  organiser and owner of the SVAMITVA scheme.
- **IIT Tirupati Navavishkar I-Hub Foundation**, **IIIT Tirupati**, and the
  **National Informatics Centre (NIC)** — hackathon partners.
- The **SVAMITVA scheme** field survey teams, for the underlying drone
  orthophoto and LiDAR data that made this project possible.

---

## License

No license file was provided with the original hackathon submission. Add a
`LICENSE` file here (MIT/Apache-2.0 are common choices for hackathon code)
before treating this repository as open for reuse.
