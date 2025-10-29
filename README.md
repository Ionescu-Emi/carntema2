# SVHN Classification Experiments with VGG-13

## Overview
I experimented with 4 configurations of VGG-13 on the SVHN dataset (100 classes), focusing on augmentations, EMA, splitting, and TTA. All achieved >60% accuracy. Key metrics: validation/test accuracy after 45-100 epochs. Experiments run on Kaggle P100 GPU.

## Configurations and Results

1. **Baseline VGG-13 with Mixup/Cutmix & RandAugment**  
   - Key: RandAugment (14 ops), mixup/cutmix (alpha=0.2), label smoothing=0.05, OneCycleLR, no EMA. Batch=128, epochs=100.  
   - Val Acc: 74.50% (best at epoch 90). Test Acc: 74.49%.  
   - [Code Link](https://www.kaggle.com/code/emi0011/atnn-2025-competition-2-baseline?scriptVersionId=271530837)  
   - Strong due to diverse augmentations balancing under/overfitting.

2. **VGG-13 with EMA, TrivialAugmentWide & Gradient Clipping**  
   - Key: Added EMA (0.995), TrivialAugmentWide, grad clip=1.0, computed stats on train, label smoothing=0.1, higher LR=1.8e-3. Batch=128, epochs=100.  
   - Val Acc: 73.85% (EMA best). Test Acc: 73.60%.  
   - [Code Link](https://www.kaggle.com/code/emi0011/atnn-2025-competition-2-baseline?scriptVersionId=271630850)  
   - EMA smoothed weights for stability, but higher smoothing reduced peak vs. baseline (tradeoff in variance).

3. **VGG-13 with Stratified Split & ColorJitter**  
   - Key: Stratified train/val split (90/10), ColorJitter (0.15 params), no mixup, computed split stats, lower epochs=45. Batch=128.  
   - Val Acc: 69.56% (on split). Test Acc: 70.53%.  
   - [Code Link](https://www.kaggle.com/code/emi0011/notebook4756e35c17/notebook?scriptVersionId=271713798)  
   - Split ensured balanced classes, but fewer augs led to overfitting; lower acc vs. full-train baselines.

4. **VGG-13 with Selective TTA at Inference**  
   - Key: Baseline augs + TTA (5x: flips/rotations) only at submission, no TTA in val. Batch=256 for TTA. Epochs=100.  
   - Val Acc: 74.19% (no TTA). Test Acc: 73.20% (with TTA).  
   - [Code Link](https://www.kaggle.com/code/emi0011/notebook66ab8f61fa)  
   - TTA boosted hard samples (+~1-2% on test), but val without TTA underestimated; rotations helped rotation-invariant features.

## Analysis
Baseline (1) excelled with aggressive augs preventing overfitting. EMA (2) stabilized but slightly lowered peaks due to averaging. Split (3) improved class balance but reduced train data/augs, dropping acc ~4-5%. TTA (4) enhanced inference robustness without train overhead, gaining on ambiguous images. Differences stem from aug diversity (higher=better generalization) vs. stability (EMA/TTA). Best: Config 1 (74.5%), as TTA adds compute without proportional val gains.
