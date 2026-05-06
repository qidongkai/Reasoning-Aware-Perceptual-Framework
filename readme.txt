 RAPF: Reasoning-Aware Perceptual Framework
 Open-Set Wild Plant Image Segmentation in Natural Complex Scenes

This is the full official implementation of the RAPF model for open-set wild plant segmentation and recognition.

 1. Overview
RAPF is a reasoning-aware perceptual framework that supports open-set recognition of wild plant species in complex natural scenes. It uses a five-stage closed-loop pipeline:
Perception ¡ú Retrieval ¡ú Reasoning ¡ú Decision ¡ú Iteration.

 2. Key Features
- Open-set recognition for known and unknown wild plant species
- CLIP-DINOv2 fused visual feature extraction
- High-quality instance segmentation using HQ-SAM
- Botanical knowledge graph reasoning (WildPlantKG)
- Dempster-Shafer (D-S) evidence fusion for interpretable results
- End-to-end training, inference, and evaluation

 3. Performance
- mIoU (Known Categories): 89.2%
- F1 Score (Unknown Species): 84.7%
- AUROC: 0.93
- Mean F1: 87.0%

 4. Dataset: WildPlantOpenSet-10K
- Total images: 10,240
- Known species: 612
- Unknown species: 527
- Annotations: instance masks, species labels, family/genus taxonomy, habitat information
- Scenes: forests, shrubs, grasslands, wetlands, alpine meadows, occlusions, uneven lighting, multi-species coexistence

**Dataset Download Link:**
https://pan.quark.cn/s/e937a5798aba

 5. Pretrained Model
HQ-SAM checkpoint is required for mask generation.
Download link:
https://huggingface.co/lkeab/hq-sam/resolve/main/hq_sam_vitl.pth

Place the file in:
checkpoints/hq_sam_vitl.pth

 6. Project Structure
configs/                Model and training configurations
data/                   Dataset loader, transforms, knowledge graph
models/                 Core RAPF modules
engine/                 Trainer and evaluator
utils/                  Tools, metrics, logging
train.py                Training script
test.py                 Inference and evaluation script
requirements.txt        Dependencies
README.txt              This document

 7. Environment Setup
Install all required packages:
pip install -r requirements.txt

 8. Data Preparation
1£©. Download WildPlantOpenSet-10K.zip from the provided link
2£©{}. Extract the dataset to:
   data/WildPlantOpenSet-10K/
3£©. Place the HQ-SAM checkpoint in:
   checkpoints/hq_sam_vitl.pth

 9. Usage

 Train the model
python train.py

 Test on a single image
Place your image as test.jpg
python test.py

 Evaluate on full test set
python test.py --eval

 10. Outputs
- Species prediction (or "unknown")
- Confidence score
- Instance segmentation mask
- Top-5 similar species from knowledge graph
- Visual result image
- Structured JSON report

 11. Citation
If you use this code or dataset, please cite:
Qi, D., Lim, C. S., & V. Sivakumar. Reasoning-Aware Perceptual Framework for Open-Set Wild Plant Image Segmentation in Natural Complex Scenes.

 12. Contact
Corresponding author: TP086393@mail.apu.edu.my
