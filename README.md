# Crop-Related Machine Learning and Deep Learning Projects
### Summer Internship 2024 | Indian Institute of Engineering Science and Technology (IIEST), Shibpur

A collection of Machine Learning and Deep Learning projects undertaken during my **Summer Internship 2024 at IIEST Shibpur**, covering agricultural crop recommendation, crop yield prediction, and pest classification.

The repository is organized into three major components:

- **Crop Recommendation** – Deep Learning-based multiclass crop recommendation using Indian agricultural data.
- **Crop Yield Prediction** – Machine Learning and predictive modelling for agricultural yield estimation.
- **Pest Classification** – Image-based classification of agricultural pests.

---
## 1. Crop Recommendation System

**ICRS-ANN: AI-Driven Crop Recommendation System for Indian Agricultural Sector with SMOTE-ENN Based Balancing and TSPE Optimization**

This project explores a Deep Learning-based approach to agricultural crop recommendation using historical Indian agricultural data. The proposed Artificial Neural Network (ANN) model predicts suitable crops using soil characteristics, climatic conditions, and agricultural production attributes.

### Methodology and Implementation

- **Dataset:** 99,849 records spanning 28 years of agricultural data, covering 51 crop classes across Indian states and union territories.
- **Data Preprocessing:** Missing-value assessment, IQR-based outlier detection, logarithmic transformation, categorical encoding, feature standardization, correlation analysis, and PCA-based dimensionality reduction.
- **Class Imbalance Handling:** SMOTE-ENN (Synthetic Minority Oversampling Technique – Edited Nearest Neighbours) with random sampling.
- **Deep Learning Architecture:** Multilayer Perceptron (MLP) with three hidden layers (128, 64, 32 neurons), ReLU activation, Adam optimizer, and Cross-Entropy Loss.
- **Hyperparameter Optimization:** Tree-Structured Parzen Estimator (TPE) for Bayesian hyperparameter optimization.
- **Evaluation:** 10-fold cross-validation using accuracy, precision, recall, F1-score, specificity, and ROC-AUC.

### Performance

| Evaluation Metric | Result |
|---|---:|
| Average Accuracy | 99.21% |
| Specificity | 99.97% |
| Validation Strategy | 10-fold Cross-Validation |

The optimized pipeline improved average accuracy from 96.51% to 99.21% following the application of SMOTE-ENN and random sampling.

### Technical Report

The work is documented in a detailed technical report written in a **research publication-style format**, comprising an abstract, introduction, literature review, methodology, implementation, experimental evaluation, comparative analysis, and conclusion.

**Important clarification:** This document is an independently prepared technical report in publication style. It has not been submitted to any journal or conference, published, or uploaded to any external publication or research repository.

📄 **[Read the Crop Recommendation Technical Report](Crop_Recommendation_Actual.pdf)**

---

## 2. Crop Yield Prediction

This component focuses on applying Machine Learning and predictive modelling techniques to agricultural crop yield estimation.

### Results and Visualizations

The following visualizations showcase the experimental results and outputs obtained during the project.

<!-- BEGIN CROP YIELD GALLERY -->
<!-- END CROP YIELD GALLERY -->

---

## 3. Pest Classification

This component explores image-based agricultural pest classification using Machine Learning / Deep Learning techniques.

### Results and Visualizations

The following visualizations showcase the classification results and experimental outputs.

<!-- Add the PNG images from Pest_Classification/Results/ below -->

<!-- BEGIN PEST CLASSIFICATION GALLERY -->
<!-- END PEST CLASSIFICATION GALLERY -->

---

## Repository Overview

| Component | Domain | Primary Objective |
|---|---|---|
| Crop Recommendation | Deep Learning | Multiclass crop recommendation |
| Crop Yield Prediction | Machine Learning | Agricultural yield prediction |
| Pest Classification | Computer Vision | Agricultural pest classification |

---

## Internship

**Summer Internship 2024**  
Indian Institute of Engineering Science and Technology (IIEST), Shibpur, West Bengal, India.

The repository consolidates the project work carried out during the internship, including implementations, experimental results, and technical documentation.
