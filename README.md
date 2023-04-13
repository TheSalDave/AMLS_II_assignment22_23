# Diabetic Retinopathy Risk Assessment using CNNs

This repository contains the work done for the Applied Machine Learning Systems 2 module at UCL on the topic of Diabetic Retinopathy (DR) risk assessment using Convolutional Neural Networks (CNNs). 

## Overview

DR is a leading cause of vision loss and blindness worldwide. Early detection of DR can improve the quality of life of affected patients. The aim of this project was to develop a DR risk classifier using CNNs trained on the <a href="https://www.kaggle.com/c/aptos2019-blindness-detection">APTOS 2019 Blindness Detection dataset from Kaggle</a>. An image pre-processing pipeline was developed to crop uninformative areas and filter out noise. Multiple CNNs were trained, evaluated, and compared using 60/20/20 split, leveraging image augmentations, hyperparameter tuning, and transfer learning to enhance performance. The best performing model was RegnetY16GF with a score of 73.3% accuracy on the unseen dataset, which is comparable to other works from the literature review and <a href="https://www.kaggle.com/c/aptos2019-blindness-detection">competitors from Kaggle</a>.

## Repository Structure

The repository is structured as follows:

- `Datasets/`: The datasets from the competition should be inserted into this folder.
- `DataPrep/`: The development of the pre-processing pipeline is shown in this folder.
- `Models/`: The model training process with hyperparameter tuning can be found in this folder.
- `Figures/`: All relevant figures are saved in this folder.
- `main.py`: Contains the complete automated workflow and can be run from the terminal using "python3 main.py". All methods and classes in `main.py` are provided with <a href="https://peps.python.org/pep-0257/">docstrings and type annotations</a> for better code quality.

## Model Performance

The following table shows the performance of the different versions of the models:

| Model               | Train Acc | Val Acc  | Test Acc |
|---------------------|-----------|----------|----------|
| CNN (bs 64 lr 0.001)     | 66.36%    | 62.50%   |          |
| AlexNet (bs 64 lr 0.001) | 60.89%    | 60.09%   |          |
| Resnet50 (bs 64 lr 0.001)| 46.32%    | 48.58%   |          |
| RegnetY16GF (bs 32 lr 0.001) | 69.46% | 70.74% |          |
| RegnetY16GF (bs 64 lr 0.001) | 71.65% | 70.60% |          |
| RegnetY16GF (bs 64 lr 0.0001) | 72.61% | 71.16% |          |
| RegnetY16GF (bs 64 lr MIX)   | 75.37% | 71.88% | 73.3%    |
| EfficientNetB7 (bs 32 lr 0.001) | 69.14% | 65.06% |          |
| EfficientNetB7 (bs 64 lr 0.001) | 69.53% | 65.9%  |          |
| EfficientNetB7 (bs 64 lr 0.0001) | 72.61% | 71.16% |          |
| EfficientNetB7 (bs 64 lr MIX)   | 66.89% | 63.49% |          |

## Future Work

Although the proposed DR risk classifier using CNNs achieved a relatively high accuracy score, there is still room for improvement. The following are some of the potential future work that can be done to enhance the performance of the model:

 - Perform a thorough grid search sweep using WandB for the hyperparameters listed in the hyperparameter tuning section to find the optimal set of hyperparameters for the model. This can further enhance the model's performance.

 - Explore other pre-trained models such as VGG-16, DenseNet, and GoogLeNet to see if they can lead to higher prediction accuracy.

 - Combine the small dataset with large external ones to improve the predictions. This approach was done in the literature review and by Kaggle competitors, and it has been shown to enhance the predictions.

 - Use ensembling techniques to combine the prediction of multiple models. This approach can be useful in situations where no single model can provide the best prediction accuracy.

 - Apply pseudo labelling technique to retrain the model on an enlarged dataset. This approach can be done iteratively, and it has been shown to lead to better scores with every iteration. The Kaggle competition provided a relatively large test set without any labels that can be used for this purpose.

 - Apply explainability techniques such as saliency maps to analyze and segment the final model. This can improve the model's reliability and offer more insight into regions of interest when analyzing retinal images.

By implementing these potential future works, the proposed DR risk classifier using CNNs can be improved to reliably assess DR risk in patients.

