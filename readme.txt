Reasoning-Aware-Perceptual-Framework-RAPF-
The Reasoning-Aware Perceptual Framework (RAPF) is an open-set wild plant segmentation model using a closed-loop perception-reasoning structure. It fuses CLIP-DINOv2 features, applies HQ-SAM for masking, and uses Dempster-Shafer reasoning with botanical knowledge to identify known and unknown species reliably in complex natural scenes. To support end-side deployment and low-data adaptation, RAPF integrates the Self-Distillation and Dynamic Inference Vision Transformer (SDViT) and the Unsupervised Domain Adaptation Distillation Module (UDAM).

This is the full official implementation of the RAPF model for open-set wild plant segmentation and recognition.

Overview RAPF is a reasoning-aware perceptual framework that supports open-set recognition of wild plant species in complex natural scenes. It uses a five-stage closed-loop pipeline: Perception → Retrieval → Reasoning → Decision → Iteration. Within this pipeline, SDViT performs dynamic token elimination and self-distillation compensation for low-latency visual perception, while UDAM enables cross-domain knowledge transfer with unlabeled target data to reduce pixel-level annotation dependence and privacy risks.

Key Features

Open-set recognition for known and unknown wild plant species

CLIP-DINOv2 fused visual feature extraction

SDViT-based self-distillation and dynamic inference for efficient end-side perception

UDAM-based unsupervised domain adaptation distillation for low-data, privacy-safe cross-domain transfer

High-quality instance segmentation using HQ-SAM

Botanical knowledge graph reasoning (WildPlantKG)

Dempster-Shafer (D-S) evidence fusion for interpretable results

End-to-end training, inference, and evaluation

Integrated Distillation Techniques: SDViT and UDAM
SDViT (Self-Distillation and Dynamic Inference Vision Transformer) is designed for AR end-side deployment and adopted in RAPF's perception stage to sustain low-latency, high-accuracy feature extraction. Its core mechanism is dynamic token elimination: redundant visual tokens are progressively identified and pruned during inference, so that computation scales down with scene complexity rather than being fixed. To prevent the accuracy loss that normally accompanies token reduction, SDViT introduces self-distillation compensation, in which the full-token representation supervises the pruned pathway within the same network, preserving fine-grained parsing capability under latency constraints. Combined with self-supervised pre-training, SDViT maintains strong generalization when labeled data are scarce. In RAPF, this allows the CLIP-DINOv2 fused feature extraction and subsequent HQ-SAM masking to run efficiently on resource-limited terminals without sacrificing the pixel-level precision required for instance masks.

UDAM (Unsupervised Domain Adaptation Distillation Module) supports cross-domain knowledge transfer using pre-trained parameters and unlabeled target data, and is used in RAPF to adapt the model to new habitats, imaging conditions and species distributions. UDAM distills knowledge from a source-domain teacher into the target-domain student without requiring pixel-level manual annotations on the target data, thereby reducing labeling costs and mitigating privacy risks associated with collecting and annotating sensitive field imagery. This is particularly relevant for wild plant scenes, where domain shifts arise from forests, wetlands, alpine meadows, uneven lighting and occlusion, and where expert annotation is expensive. UDAM enables RAPF to transfer botanical and visual knowledge across such domains while keeping the reasoning and D-S evidence fusion stages consistent and interpretable.

Together, SDViT and UDAM alleviate the inherent efficiency-accuracy contradiction and high data reliance of traditional distillation methods, providing feasible engineering guidance for the large-scale deployment of RAPF on wearable AR and vehicle perception terminals, and offering an efficient distillation optimization solution for resource-limited, privacy-sensitive edge intelligent scenarios.

Performance

mIoU (Known Categories): 89.2%

F1 Score (Unknown Species): 84.7%

AUROC: 0.93

Mean F1: 87.0%

Dataset: WildPlantOpenSet-10K

Total images: 10,240

Known species: 612

Unknown species: 52

Annotations: instance masks, species labels, family/genus taxonomy, habitat information

Scenes: forests, shrubs, grasslands, wetlands, alpine meadows, occlusions, uneven lighting, multi-species coexistence

Dataset Download Link: doi:10.5281/zenodo.20183969

Pretrained Model HQ-SAM checkpoint is required for mask generation. Download link: https://huggingface.co/lkeab/hq-sam/resolve/main/hq_sam_vitl.pth
Place the file in: checkpoints/hq_sam_vitl.pth

Project Structure configs/ Model and training configurations data/ Dataset loader, transforms, knowledge graph models/ Core RAPF modules (including SDViT and UDAM) engine/ Trainer and evaluator utils/ Tools, metrics, logging train.py Training script test.py Inference and evaluation script requirements.txt Dependencies README.txt This document

Environment Setup Install all required packages: pip install -r requirements.txt

Data Preparation 1）. Download WildPlantOpenSet-10K.zip from the provided link 2）{}. Extract the dataset to: data/WildPlantOpenSet-10K/ 3）. Place the HQ-SAM checkpoint in: checkpoints/hq_sam_vitl.pth

Usage

Train the model python train.py

Test on a single image Place your image as test.jpg python test.py

Evaluate on full test set python test.py --eval

Outputs

Species prediction (or "unknown")

Confidence score

Instance segmentation mask

Top-5 similar species from knowledge graph

Visual result image

Structured JSON report

Citation If you use this code or dataset, please cite: Qi, D., Lim, C. S., & V. Sivakumar. Reasoning-Aware Perceptual Framework for Open-Set Wild Plant Image Segmentation in Natural Complex Scenes.

Contact Corresponding author: TP086393@mail.apu.edu.my
