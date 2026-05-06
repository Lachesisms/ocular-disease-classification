# ocular-disease-classification

8-class retinal fundus disease classifier on ODIR-5K | 
86.8% accuracy | ResNet50, EfficientNetB0, ResNet152

## Overview
Multi-class fundus image classifier trained on the ODIR-5K 
dataset to detect 8 ocular diseases including Diabetes, 
Glaucoma, Cataract, AMD, Hypertension, Pathological Myopia, 
and Normal/Other categories.

## Results
| Model          | Accuracy | F1    |
|----------------|----------|-------|
| ResNet50       | 86.8%    | 0.868 |
| EfficientNetB0 | -        | -     |
| ResNet152      | -        | -     |

## Key Features
- Two-phase transfer learning strategy
- Focal Loss + per-class sample weighting for class imbalance
- Grad-CAM saliency visualization for model interpretability
- Flask web application for end-to-end inference

## Requirements
