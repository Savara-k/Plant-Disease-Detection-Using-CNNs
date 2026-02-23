# Plant-Disease-Detection-Using-CNNs

RGB vs. Grayscale vs. Edge-Based Image Representations

A machine learning study evaluating how different image preprocessing techniques affect Convolutional Neural Network (CNN) performance in plant disease detection.
This project compares three input representations
* RGB color images
* grayscale images
* edge-detected
images using the PlantVillage dataset to determine which visual features are most critical for accurate disease classification.

---

### Executive Summary
Early detection of plant diseases is crucial for agriculture, especially where expert diagnosis is unavailable. While CNNs can automate disease detection, it is unclear how much visual information can be removed from images before model performance degrades.

This project investigates whether CNNs require:

* Full color information
* Texture and intensity
* Structural outlines

to accurately detect plant diseases.

Three identical CNN models were trained on different versions of the same dataset:
* RGB images
* Grayscale images
* Edge-detected images
  
Results show that removing color and texture significantly reduces model accuracy, highlighting the importance of chromatic cues such as yellowing, spotting, and discoloration.

---
## Research Objective

* To determine which visual features — color, texture, or shape — are most important for CNN-based plant disease detection.
Dataset
* PlantVillage Dataset (Kaggle)
* Over 20,000 leaf images across 15 crops and 38 diseases

---
## Reorganized into a binary classification task:

* Classification Task
* Healthy
* Diseased
* A held-out test set of 4,128 images was reserved for evaluation to ensure generalization to unseen data.
---

## Methodology:

* Preprocessing
* Images resized to uniform resolution
* Pixel normalization to 0–1 range
* Identical train/validation splits across experiments

---
## Model Architectures:

All experiments used the same CNN architecture to isolate the effect of input representation.

1 — RGB CNN (Baseline)
* Full three-channel color input
* Preserves color, texture, and spatial information
* Binary classification with sigmoid activation

2 — Grayscale CNN
* Single-channel grayscale conversion
* Removes color cues
* Tests reliance on texture and intensity patterns

3 — Edge-Detection CNN
* Edge maps generated using classical edge detection
* Emphasizes shape and structural boundaries
* Removes color and internal texture information
* Experimental Setup
* Implemented in TensorFlow and Keras
* Same number of epochs for all models
* Identical optimization settings

---
## Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1-score
These metrics provide a balanced assessment, particularly for detecting diseased samples.

---

## Results:

## RGB CNN Performance (Best)

* Training Accuracy: ~95%
* Validation Accuracy: ~93%
* Weighted F1-score: ~0.93

The RGB model consistently achieved the highest performance, indicating that color information is critical for disease detection. Many plant diseases produce distinct chromatic patterns such as yellowing, brown lesions, and discoloration.

## Grayscale CNN Performance

* Training Accuracy: ~91%
* Validation Accuracy: ~86%
* Weighted F1-score: ~0.86

While still effective, removing color information reduced performance. The model relied on texture and intensity patterns but struggled with subtle disease signals.

## Edge-Detection CNN Performance (Weakest)

* Training Accuracy: ~94%
* Validation Accuracy: ~79%
* F1-score: ~0.79

The large gap between training and validation accuracy indicates overfitting. With only structural outlines available, the model memorized shapes rather than learning disease features.
This demonstrates that leaf structure alone is insufficient for accurate diagnosis.

Comparative Performance

---

## Performance hierarchy:

RGB CNN > Grayscale CNN > Edge-Detection CNN
Simplifying image data reduces diagnostic information and harms model performance in agricultural tasks.

## Key Insights

* Color cues are the most important features for plant disease detection
* Texture contributes but cannot replace color
* Shape alone is insufficient
* Removing information to simplify data can reduce model effectiveness
 
CNNs require rich visual detail to capture disease-specific patterns.

---

## Conclusion
This study demonstrates that preserving color and texture is essential for CNN-based plant disease detection. Simplified representations such as grayscale or edge maps remove critical diagnostic features and reduce accuracy.
The RGB model achieved the strongest performance, confirming that chromatic information is central to identifying plant diseases.

---

## Future Work
* Evaluate models on real-world field images
* Apply transfer learning (ResNet, EfficientNet)
* Experiment with heavy data augmentation
* Improve robustness under varying lighting and environmental conditions

--- 

## Technical Stack
* Python
* TensorFlow / Keras
* NumPy / Pandas
* Matplotlib

---
## What This Project Demonstrates
* CNN implementation for image classification
* Controlled experimental design
* Comparative analysis of preprocessing techniques
  
Model evaluation using multiple metrics
Understanding of feature importance in deep learning
