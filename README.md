![image](https://github.com/user-attachments/assets/12c7c006-d23a-4881-bf4c-f7884df4e989)# Kidney-disease-detection-
This project leverages artificial intelligence to facilitate kidney disease diagnosis. A Convolutional Neural Network (CNN) was implemented with several optimizations that significantly enhanced the model’s accuracy. As a result, the model achieved an impressive 98.84% accuracy in classification.
model consists of multiple convolutional layers followed by batch normalization, max pooling, and fully connected (dense) layers. The key features include:

Convolutional Layers: Extract spatial features from images.
Batch Normalization: Stabilizes training by normalizing layer inputs.
Max Pooling: Reduces spatial dimensions to retain key features.
Global Max Pooling: Replaces flattening for better feature extraction.
Dense Layers: Fully connected layers for classification.
Softmax Activation: Outputs probabilities for four classes.    
![image](https://github.com/user-attachments/assets/e62f8a8e-547b-45b4-a449-df82b8e2beab)
Here's a visual representation of your CNN model architecture. It clearly shows the sequence of layers from input to output, including convolutional layers, batch normalization, pooling, and dense layers.


![Screenshot 2025-02-25 112135](https://github.com/user-attachments/assets/ca8b91af-ee6a-4da8-bd22-3ef739ae2f0a)
This confusion matrix evaluates the performance of your kidney disease classification model. It shows how well your model predicts each class compared to the actual labels.
Class 0 is well-classified with 1010 correct predictions and very few misclassifications.
Class 1 has 742 correct predictions and no misclassification into other classes.
Class 2 has 451 correct predictions, with 6 samples misclassified (4 into Class 0, 2 into Class 1).
Class 3 has 258 correct predictions, but 17 samples were wrongly classified as Class 1, indicating a small issue with class differentiation.


![Screenshot 2025-02-25 112127](https://github.com/user-attachments/assets/1760d2a8-87f0-49f4-8d2b-26edaef00ab7)

This study evaluates the performance of a CNN model designed for kidney disease classification. The model incorporates optimizations such as Leaky ReLU activation, Batch Normalization, and Global Max Pooling, achieving an impressive 98.84% accuracy.

1️- Confusion Matrix Analysis
The confusion matrix indicates high classification accuracy, with most predictions falling on the diagonal, meaning correct classifications. Class 3 (Kidney Disease Type 3) shows slight misclassification into Class 1, which could be improved with additional fine-tuning or data balancing.

2️- Training & Validation Performance
The accuracy plot shows that training and validation accuracy increase steadily, stabilizing near 99%, indicating effective learning. The loss plot demonstrates a rapid decline, confirming that the model minimizes errors efficiently. The small gap between training and validation curves suggests that overfitting is well-controlled.


