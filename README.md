# Brain Tumor Classification Using CNNs and Transfer Learning

### Overview

This project demonstrates how to classify brain MRI images into four categories using Convolutional Neural Networks (CNNs) and transfer learning with a pre-trained ResNet50 model. The dataset includes MRI scans of human brains categorized into **glioma**, **meningioma**, **no tumor**, and **pituitary tumor**.

### **Project Overview**

- Preprocess brain MRI images by cropping the brain region and resizing to a uniform size.
- Train a CNN model using transfer learning with a pre-trained ResNet50 backbone.
- Improve classification performance through data augmentation and fine-tuning.
- Evaluate the model using classification metrics like accuracy, precision, recall, and F1-score.
- Use Explainable AI techniques (LIME and SHAP) to visualize and interpret model predictions.

### **Dataset**

This project uses a combination of datasets from the following sources:

1. **Figshare**
2. **SARTAJ dataset**
3. **Br35H dataset**

The dataset consists of:

- **7,023 MRI images** of human brains.
- **4 Classes**:
  - **Glioma**
  - **Meningioma**
  - **No Tumor** (images taken from the Br35H dataset)
  - **Pituitary Tumor**

The images are organized into separate folders for training and testing:

- **Training Dataset**: 5,712 images
- **Testing Dataset**: 1,311 images

### Project Workflow
##### **1. Data Preprocessing**
- Crop the brain region by detecting contours in MRI scans.
- Resize the cropped images to **256x256 pixels**.
- Normalize pixel values to the range `[0, 1]` for training and testing.

##### **2. Data Augmentation**
- Apply augmentation to the training data to improve model generalization:
  - Rotation up to 30°
  - Horizontal and vertical shifts (up to 20%)
  - Zooming and shearing
  - Horizontal flipping

##### **3. Transfer Learning with ResNet50**
- Use ResNet50 pre-trained on ImageNet as a feature extractor.
- Fine-tune the last 20 layers of the ResNet50 model.
- Add custom layers:
  - **GlobalAveragePooling2D** for flattening feature maps.
  - Dense layers with ReLU activation and dropout regularization.
  - Softmax output layer for 4-class classification.

##### **4. Training**
- Train the model using the **Adam optimizer** with a learning rate of `0.0001`.
- Use a weighted data generator to handle class imbalance.

###### **5. Evaluation**
- Evaluate the model on the test set:
  - **Accuracy**: Achieved 69.26%.
  - **Classification Report**:
    - Glioma: Precision = 54%, Recall = 79%
    - Meningioma: Precision = 59%, Recall = 35%
    - No Tumor: Precision = 80%, Recall = 93%
    - Pituitary: Precision = 87%, Recall = 63%
  - **Confusion Matrix**: Visualized with `seaborn`.

##### **6. Explainable AI**
- Use **SHAP (SHapley Additive exPlanations)** to interpret model predictions.
- Use **LIME (Local Interpretable Model-agnostic Explanations)** to highlight regions contributing to predictions.

### **Key Results**

- **Model Architecture**:
  - Total Parameters: 23.85M
  - Trainable Parameters: 9.19M
  - Non-Trainable Parameters: 14.65M
- **Test Accuracy**: **69.26%**
- **Visualization**:
  - Training History: Plotted accuracy and loss over epochs.
  - Confusion Matrix: Visualized classification performance across all classes.

### Source

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
