# Heterogeneous-Convolution-for-HAR
### This is the code for [Human activity recognition using wearable sensors by heterogeneous convolutional neural networks[J]. Expert Systems with Applications, 2022: 116764.](https://www.sciencedirect.com/science/article/pii/S0957417422002299)

![](https://github.com/Chaolei98/Heterogeneous-Convolution/blob/main/overview.png)

### Here shows the model architecture and the simplfied TRAIN/VALIDATION process on PAMAP2 dataset. Experiments on other datasets are just diverse hyper-parameters as examples. If there are mistakes, welcome to point out!

## Requirements in this work
● Python 3.8.10  
● PyTorch 1.8.2 + cu111  
● Numpy 1.21.2  
● Scikit-learn  

## Train
Get required dataset from UCI Machine Learning Repository(http://archive.ics.uci.edu/ml/index.php), do data pre-processing by sliding window strategy and split the data into different sets
```
$ cd Heterogeneous-Convolution
$ python main.py
```

## Cited
If you have any questions about details, let me know. If you find it helpfutl in your work, feel free to quote
```
@article{han2022human,
  title={Human activity recognition using wearable sensors by heterogeneous convolutional neural networks},
  author={Han, Chaolei and Zhang, Lei and Tang, Yin and Huang, Wenbo and Min, Fuhong and He, Jun},
  journal={Expert Systems with Applications},
  pages={116764},
  year={2022},
  publisher={Elsevier}
}
```
