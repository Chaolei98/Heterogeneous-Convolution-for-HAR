# Heterogeneous-Convolution-for-HAR
### This is the code for [Human Activity Recognition Using Wearable Sensors by Heterogeneous Convolutional Neural Networks](https://www.sciencedirect.com/science/article/pii/S0957417422002299)

![](https://github.com/Chaolei98/Heterogeneous-Convolution/blob/main/overview.png)

### Here shows the model architecture and the simplfied TRAIN/VALIDATION process on PAMAP2 dataset. Experiments on other datasets are just diverse hyper-parameters as examoles. If there are mistakes, welcome to point out!

### Requirements in this work
● Python 3.8.10  
● PyTorch 1.8.2 + cu111
● Numpy 1.21.2
● Scikit-learn  

### Train
Get required dataset from UCI Machine Learning Repository(http://archive.ics.uci.edu/ml/index.php), do data pre-processing by sliding window strategy and split the data into training and test sets
```
$ cd Heterogeneous-Convolution-for-HAR
$ python main.py
```
