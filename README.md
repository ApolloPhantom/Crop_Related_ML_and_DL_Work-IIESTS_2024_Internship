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

### Bi-GRU_1_ev

![Bi-GRU_1_ev](Crop_Yield_Prediction/Results/Bi-GRU_1_ev.png)

### Bi-GRU_1_loss

![Bi-GRU_1_loss](Crop_Yield_Prediction/Results/Bi-GRU_1_loss.png)

### Bi-GRU_1_mae

![Bi-GRU_1_mae](Crop_Yield_Prediction/Results/Bi-GRU_1_mae.png)

### Bi-GRU_1_mse

![Bi-GRU_1_mse](Crop_Yield_Prediction/Results/Bi-GRU_1_mse.png)

### Bi-GRU_1_pred

![Bi-GRU_1_pred](Crop_Yield_Prediction/Results/Bi-GRU_1_pred.png)

### Bi-GRU_1_r2

![Bi-GRU_1_r2](Crop_Yield_Prediction/Results/Bi-GRU_1_r2.png)

### Bi-GRU_2_ev

![Bi-GRU_2_ev](Crop_Yield_Prediction/Results/Bi-GRU_2_ev.png)

### Bi-GRU_2_loss

![Bi-GRU_2_loss](Crop_Yield_Prediction/Results/Bi-GRU_2_loss.png)

### Bi-GRU_2_mae

![Bi-GRU_2_mae](Crop_Yield_Prediction/Results/Bi-GRU_2_mae.png)

### Bi-GRU_2_mse

![Bi-GRU_2_mse](Crop_Yield_Prediction/Results/Bi-GRU_2_mse.png)

### Bi-GRU_2_pred

![Bi-GRU_2_pred](Crop_Yield_Prediction/Results/Bi-GRU_2_pred.png)

### Bi-GRU_2_r2

![Bi-GRU_2_r2](Crop_Yield_Prediction/Results/Bi-GRU_2_r2.png)

### Bi-GRU_3_ev

![Bi-GRU_3_ev](Crop_Yield_Prediction/Results/Bi-GRU_3_ev.png)

### Bi-GRU_3_loss

![Bi-GRU_3_loss](Crop_Yield_Prediction/Results/Bi-GRU_3_loss.png)

### Bi-GRU_3_mae

![Bi-GRU_3_mae](Crop_Yield_Prediction/Results/Bi-GRU_3_mae.png)

### Bi-GRU_3_mse

![Bi-GRU_3_mse](Crop_Yield_Prediction/Results/Bi-GRU_3_mse.png)

### Bi-GRU_3_pred

![Bi-GRU_3_pred](Crop_Yield_Prediction/Results/Bi-GRU_3_pred.png)

### Bi-GRU_3_r2

![Bi-GRU_3_r2](Crop_Yield_Prediction/Results/Bi-GRU_3_r2.png)

### Bi-GRU_loss

![Bi-GRU_loss](Crop_Yield_Prediction/Results/Bi-GRU_loss.png)

### Bi-LSTM_1_ev

![Bi-LSTM_1_ev](Crop_Yield_Prediction/Results/Bi-LSTM_1_ev.png)

### Bi-LSTM_1_loss

![Bi-LSTM_1_loss](Crop_Yield_Prediction/Results/Bi-LSTM_1_loss.png)

### Bi-LSTM_1_mae

![Bi-LSTM_1_mae](Crop_Yield_Prediction/Results/Bi-LSTM_1_mae.png)

### Bi-LSTM_1_mse

![Bi-LSTM_1_mse](Crop_Yield_Prediction/Results/Bi-LSTM_1_mse.png)

### Bi-LSTM_1_pred

![Bi-LSTM_1_pred](Crop_Yield_Prediction/Results/Bi-LSTM_1_pred.png)

### Bi-LSTM_1_r2

![Bi-LSTM_1_r2](Crop_Yield_Prediction/Results/Bi-LSTM_1_r2.png)

### Bi-LSTM_2_ev

![Bi-LSTM_2_ev](Crop_Yield_Prediction/Results/Bi-LSTM_2_ev.png)

### Bi-LSTM_2_loss

![Bi-LSTM_2_loss](Crop_Yield_Prediction/Results/Bi-LSTM_2_loss.png)

### Bi-LSTM_2_mae

![Bi-LSTM_2_mae](Crop_Yield_Prediction/Results/Bi-LSTM_2_mae.png)

### Bi-LSTM_2_mse

![Bi-LSTM_2_mse](Crop_Yield_Prediction/Results/Bi-LSTM_2_mse.png)

### Bi-LSTM_2_pred

![Bi-LSTM_2_pred](Crop_Yield_Prediction/Results/Bi-LSTM_2_pred.png)

### Bi-LSTM_2_r2

![Bi-LSTM_2_r2](Crop_Yield_Prediction/Results/Bi-LSTM_2_r2.png)

### Bi-LSTM_3_ev

![Bi-LSTM_3_ev](Crop_Yield_Prediction/Results/Bi-LSTM_3_ev.png)

### Bi-LSTM_3_loss

![Bi-LSTM_3_loss](Crop_Yield_Prediction/Results/Bi-LSTM_3_loss.png)

### Bi-LSTM_3_mae

![Bi-LSTM_3_mae](Crop_Yield_Prediction/Results/Bi-LSTM_3_mae.png)

### Bi-LSTM_3_mse

![Bi-LSTM_3_mse](Crop_Yield_Prediction/Results/Bi-LSTM_3_mse.png)

### Bi-LSTM_3_pred

![Bi-LSTM_3_pred](Crop_Yield_Prediction/Results/Bi-LSTM_3_pred.png)

### Bi-LSTM_3_r2

![Bi-LSTM_3_r2](Crop_Yield_Prediction/Results/Bi-LSTM_3_r2.png)

### Bi-RNN_1_ev

![Bi-RNN_1_ev](Crop_Yield_Prediction/Results/Bi-RNN_1_ev.png)

### Bi-RNN_1_loss

![Bi-RNN_1_loss](Crop_Yield_Prediction/Results/Bi-RNN_1_loss.png)

### Bi-RNN_1_mae

![Bi-RNN_1_mae](Crop_Yield_Prediction/Results/Bi-RNN_1_mae.png)

### Bi-RNN_1_mse

![Bi-RNN_1_mse](Crop_Yield_Prediction/Results/Bi-RNN_1_mse.png)

### Bi-RNN_1_pred

![Bi-RNN_1_pred](Crop_Yield_Prediction/Results/Bi-RNN_1_pred.png)

### Bi-RNN_1_r2

![Bi-RNN_1_r2](Crop_Yield_Prediction/Results/Bi-RNN_1_r2.png)

### Bi-RNN_2_ev

![Bi-RNN_2_ev](Crop_Yield_Prediction/Results/Bi-RNN_2_ev.png)

### Bi-RNN_2_loss

![Bi-RNN_2_loss](Crop_Yield_Prediction/Results/Bi-RNN_2_loss.png)

### Bi-RNN_2_mae

![Bi-RNN_2_mae](Crop_Yield_Prediction/Results/Bi-RNN_2_mae.png)

### Bi-RNN_2_mse

![Bi-RNN_2_mse](Crop_Yield_Prediction/Results/Bi-RNN_2_mse.png)

### Bi-RNN_2_pred

![Bi-RNN_2_pred](Crop_Yield_Prediction/Results/Bi-RNN_2_pred.png)

### Bi-RNN_2_r2

![Bi-RNN_2_r2](Crop_Yield_Prediction/Results/Bi-RNN_2_r2.png)

### Bi-RNN_3_ev

![Bi-RNN_3_ev](Crop_Yield_Prediction/Results/Bi-RNN_3_ev.png)

### Bi-RNN_3_loss

![Bi-RNN_3_loss](Crop_Yield_Prediction/Results/Bi-RNN_3_loss.png)

### Bi-RNN_3_mae

![Bi-RNN_3_mae](Crop_Yield_Prediction/Results/Bi-RNN_3_mae.png)

### Bi-RNN_3_mse

![Bi-RNN_3_mse](Crop_Yield_Prediction/Results/Bi-RNN_3_mse.png)

### Bi-RNN_3_pred

![Bi-RNN_3_pred](Crop_Yield_Prediction/Results/Bi-RNN_3_pred.png)

### Bi-RNN_3_r2

![Bi-RNN_3_r2](Crop_Yield_Prediction/Results/Bi-RNN_3_r2.png)

### GRU_1_ev

![GRU_1_ev](Crop_Yield_Prediction/Results/GRU_1_ev.png)

### GRU_1_loss

![GRU_1_loss](Crop_Yield_Prediction/Results/GRU_1_loss.png)

### GRU_1_mae

![GRU_1_mae](Crop_Yield_Prediction/Results/GRU_1_mae.png)

### GRU_1_mse

![GRU_1_mse](Crop_Yield_Prediction/Results/GRU_1_mse.png)

### GRU_1_pred

![GRU_1_pred](Crop_Yield_Prediction/Results/GRU_1_pred.png)

### GRU_1_r2

![GRU_1_r2](Crop_Yield_Prediction/Results/GRU_1_r2.png)

### GRU_2_ev

![GRU_2_ev](Crop_Yield_Prediction/Results/GRU_2_ev.png)

### GRU_2_loss

![GRU_2_loss](Crop_Yield_Prediction/Results/GRU_2_loss.png)

### GRU_2_mae

![GRU_2_mae](Crop_Yield_Prediction/Results/GRU_2_mae.png)

### GRU_2_mse

![GRU_2_mse](Crop_Yield_Prediction/Results/GRU_2_mse.png)

### GRU_2_pred

![GRU_2_pred](Crop_Yield_Prediction/Results/GRU_2_pred.png)

### GRU_2_r2

![GRU_2_r2](Crop_Yield_Prediction/Results/GRU_2_r2.png)

### GRU_3_ev

![GRU_3_ev](Crop_Yield_Prediction/Results/GRU_3_ev.png)

### GRU_3_loss

![GRU_3_loss](Crop_Yield_Prediction/Results/GRU_3_loss.png)

### GRU_3_mae

![GRU_3_mae](Crop_Yield_Prediction/Results/GRU_3_mae.png)

### GRU_3_mse

![GRU_3_mse](Crop_Yield_Prediction/Results/GRU_3_mse.png)

### GRU_3_pred

![GRU_3_pred](Crop_Yield_Prediction/Results/GRU_3_pred.png)

### GRU_3_r2

![GRU_3_r2](Crop_Yield_Prediction/Results/GRU_3_r2.png)

### LSTM_1_ev

![LSTM_1_ev](Crop_Yield_Prediction/Results/LSTM_1_ev.png)

### LSTM_1_loss

![LSTM_1_loss](Crop_Yield_Prediction/Results/LSTM_1_loss.png)

### LSTM_1_mae

![LSTM_1_mae](Crop_Yield_Prediction/Results/LSTM_1_mae.png)

### LSTM_1_mse

![LSTM_1_mse](Crop_Yield_Prediction/Results/LSTM_1_mse.png)

### LSTM_1_pred

![LSTM_1_pred](Crop_Yield_Prediction/Results/LSTM_1_pred.png)

### LSTM_1_r2

![LSTM_1_r2](Crop_Yield_Prediction/Results/LSTM_1_r2.png)

### LSTM_2_ev

![LSTM_2_ev](Crop_Yield_Prediction/Results/LSTM_2_ev.png)

### LSTM_2_loss

![LSTM_2_loss](Crop_Yield_Prediction/Results/LSTM_2_loss.png)

### LSTM_2_mae

![LSTM_2_mae](Crop_Yield_Prediction/Results/LSTM_2_mae.png)

### LSTM_2_mse

![LSTM_2_mse](Crop_Yield_Prediction/Results/LSTM_2_mse.png)

### LSTM_2_pred

![LSTM_2_pred](Crop_Yield_Prediction/Results/LSTM_2_pred.png)

### LSTM_2_r2

![LSTM_2_r2](Crop_Yield_Prediction/Results/LSTM_2_r2.png)

### LSTM_3_ev

![LSTM_3_ev](Crop_Yield_Prediction/Results/LSTM_3_ev.png)

### LSTM_3_loss

![LSTM_3_loss](Crop_Yield_Prediction/Results/LSTM_3_loss.png)

### LSTM_3_mae

![LSTM_3_mae](Crop_Yield_Prediction/Results/LSTM_3_mae.png)

### LSTM_3_mse

![LSTM_3_mse](Crop_Yield_Prediction/Results/LSTM_3_mse.png)

### LSTM_3_pred

![LSTM_3_pred](Crop_Yield_Prediction/Results/LSTM_3_pred.png)

### LSTM_3_r2

![LSTM_3_r2](Crop_Yield_Prediction/Results/LSTM_3_r2.png)

### RNN_1_ev

![RNN_1_ev](Crop_Yield_Prediction/Results/RNN_1_ev.png)

### RNN_1_loss

![RNN_1_loss](Crop_Yield_Prediction/Results/RNN_1_loss.png)

### RNN_1_mae

![RNN_1_mae](Crop_Yield_Prediction/Results/RNN_1_mae.png)

### RNN_1_mse

![RNN_1_mse](Crop_Yield_Prediction/Results/RNN_1_mse.png)

### RNN_1_pred

![RNN_1_pred](Crop_Yield_Prediction/Results/RNN_1_pred.png)

### RNN_1_r2

![RNN_1_r2](Crop_Yield_Prediction/Results/RNN_1_r2.png)

### RNN_2_ev

![RNN_2_ev](Crop_Yield_Prediction/Results/RNN_2_ev.png)

### RNN_2_loss

![RNN_2_loss](Crop_Yield_Prediction/Results/RNN_2_loss.png)

### RNN_2_mae

![RNN_2_mae](Crop_Yield_Prediction/Results/RNN_2_mae.png)

### RNN_2_mse

![RNN_2_mse](Crop_Yield_Prediction/Results/RNN_2_mse.png)

### RNN_2_pred

![RNN_2_pred](Crop_Yield_Prediction/Results/RNN_2_pred.png)

### RNN_2_r2

![RNN_2_r2](Crop_Yield_Prediction/Results/RNN_2_r2.png)

### RNN_3_ev

![RNN_3_ev](Crop_Yield_Prediction/Results/RNN_3_ev.png)

### RNN_3_loss

![RNN_3_loss](Crop_Yield_Prediction/Results/RNN_3_loss.png)

### RNN_3_mae

![RNN_3_mae](Crop_Yield_Prediction/Results/RNN_3_mae.png)

### RNN_3_mse

![RNN_3_mse](Crop_Yield_Prediction/Results/RNN_3_mse.png)

### RNN_3_pred

![RNN_3_pred](Crop_Yield_Prediction/Results/RNN_3_pred.png)

### RNN_3_r2

![RNN_3_r2](Crop_Yield_Prediction/Results/RNN_3_r2.png)


<!-- END CROP YIELD GALLERY -->

---

## 3. Pest Classification

This component explores image-based agricultural pest classification using Machine Learning / Deep Learning techniques.

### Results and Visualizations

The following visualizations showcase the classification results and experimental outputs.

<!-- Add the PNG images from Pest_Classification/Results/ below -->

<!-- BEGIN PEST CLASSIFICATION GALLERY -->

### Accu_AdvancedCNN

![Accu_AdvancedCNN](Pest_Classification/Results/Accu_AdvancedCNN.png)

### Accu_IntermediateCNN

![Accu_IntermediateCNN](Pest_Classification/Results/Accu_IntermediateCNN.png)

### Accu_SimpleCNN

![Accu_SimpleCNN](Pest_Classification/Results/Accu_SimpleCNN.png)

### Accu_VGG

![Accu_VGG](Pest_Classification/Results/Accu_VGG.png)

### ConfusionMatrix_AdvancedCNN

![ConfusionMatrix_AdvancedCNN](Pest_Classification/Results/ConfusionMatrix_AdvancedCNN.png)

### ConfusionMatrix_IntermediateCNN

![ConfusionMatrix_IntermediateCNN](Pest_Classification/Results/ConfusionMatrix_IntermediateCNN.png)

### ConfusionMatrix_SimpleCNN

![ConfusionMatrix_SimpleCNN](Pest_Classification/Results/ConfusionMatrix_SimpleCNN.png)

### ConfusionMatrix_VGG

![ConfusionMatrix_VGG](Pest_Classification/Results/ConfusionMatrix_VGG.png)

### Loss_AdvancedCNN

![Loss_AdvancedCNN](Pest_Classification/Results/Loss_AdvancedCNN.png)

### Loss_IntermediateCNN

![Loss_IntermediateCNN](Pest_Classification/Results/Loss_IntermediateCNN.png)

### Loss_SimpleCNN

![Loss_SimpleCNN](Pest_Classification/Results/Loss_SimpleCNN.png)

### Loss_VGG

![Loss_VGG](Pest_Classification/Results/Loss_VGG.png)

### ROC_AUC_AdvancedCNN

![ROC_AUC_AdvancedCNN](Pest_Classification/Results/ROC_AUC_AdvancedCNN.png)

### ROC_AUC_IntermediateCNN

![ROC_AUC_IntermediateCNN](Pest_Classification/Results/ROC_AUC_IntermediateCNN.png)

### ROC_AUC_SimpleCNN

![ROC_AUC_SimpleCNN](Pest_Classification/Results/ROC_AUC_SimpleCNN.png)

### ROC_AUC_VGG

![ROC_AUC_VGG](Pest_Classification/Results/ROC_AUC_VGG.png)

### RadarAdvancedCNN

![RadarAdvancedCNN](Pest_Classification/Results/RadarAdvancedCNN.png)

### RadarIntermediateCNN

![RadarIntermediateCNN](Pest_Classification/Results/RadarIntermediateCNN.png)

### RadarSimpleCNN

![RadarSimpleCNN](Pest_Classification/Results/RadarSimpleCNN.png)

### RadarVGG

![RadarVGG](Pest_Classification/Results/RadarVGG.png)


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
