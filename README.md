# README
## The Author
Liyuan Tan

## Project Name
### Sentiment Analysis
A deep neural network approach for sentiment analysis

## Build
Software Environment:
Jupyter Notebook, MATLAB R2019a (Any version should be working)

Version Control:
Python 3.6

Required Libraries:
Numpy 1.14.3, Pandas 0.23.0, Python Regular Expression Operations 2.2.1, Natural Language Toolkit 3.3, Sklearn 0.19.1, Keras 2.2.4 (Any version should be working)

## Contributor
Thanks to albertbup for the provision of the source code of deep belief network. Thanks to Prof. Guangbin Huang, Nanyang Technological University, Singapore, for the provision of the source code of deep extreme learning machine.

Datasets are downloaded from Kaggle.

## Setup
Download the code from master branch as sentiment_analysis-master.zip.

Unzip sentiment_analysis-master.zip, open the folder, there should be the 'DBN' folder, the 'DELM' folder, the 'LSTM' folder, and the 'numpy_to_matlab' folder.

Visit https://drive.google.com/drive/folders/15HwmitIGlP0c87LXAzBgAMeuzHMvyup-?usp=sharing to download the dataset folder named 'input'.

Put the 'input' folder into the 'LSTM' folder, the 'DBN' folder, and the 'numpy_to_matlab' folder.

## Run
The project includes three deep neural network approaches for sentiment analysis. Each approach is different from the others.

### Run LSTM
Open Jupyter Notebook.
Get into the 'LSTM' folder.

#### Run LSTM_movie_validate.ipynb to get the best set of the hyper parameters for implementing LSTM on the IMDB review dataset
1. You have to adjust your own hyper parameters such as 'Max words', 'Embedding size', 'Max length', 'LSTM parameter', 'Dropout rates', 'Dense nodes', 'Epochs', and 'Batch_size'.
2. You need to collect all the result after running the file.
3. You need to choose the best set of the hyper parameters for the test.
 
#### Run LSTM_amazon_validate.ipynb to get the best set of the hyper parameters for implementing LSTM on the Amazon review dataset
1. You have to adjust your own hyper parameters such as 'Max words', 'Embedding size', 'Max length', 'LSTM parameter', 'Dropout rates', 'Dense nodes', 'Epochs', and 'Batch_size'.
2. You need to collect all the result after running the file.
3. You need to choose the best set of the hyper parameters for the test.

#### Run LSTM_hotel_validate.ipynb to get the best set of the hyper parameters for implementing LSTM on the Hotel review dataset
1. You have to adjust your own hyper parameters such as 'Max words', 'Embedding size', 'Max length', 'LSTM parameter', 'Dropout rates', 'Dense nodes', 'Epochs', and 'Batch_size'.
2. You need to collect all the result after running the file.
3. You need to choose the best set of the hyper parameters for the test.

#### Run LSTM_airline_validate.ipynb to get the best set of the hyper parameters for implementing LSTM on the US airline sentiment dataset
1. You have to adjust your own hyper parameters such as 'Max words', 'Embedding size', 'Max length', 'LSTM parameter', 'Dropout rates', 'Dense nodes', 'Epochs', and 'Batch_size'.
2. You need to collect all the result after running the file.
3. You need to choose the best set of the hyper parameters for the test.

#### Run LSTM_twitter_validate.ipynb to get the best set of the hyper parameters for implementing LSTM on the Twitter dataset
1. You have to adjust your own hyper parameters such as 'Max words', 'Embedding size', 'Max length', 'LSTM parameter', 'Dropout rates', 'Dense nodes', 'Epochs', and 'Batch_size'.
2. You need to collect all the result after running the file.
3. You need to choose the best set of the hyper parameters for the test.

#### Run LSTM_reddit_validate.ipynb to get the best set of the hyper parameters for implementing LSTM on the Reddit dataset
1. You have to adjust your own hyper parameters such as 'Max words', 'Embedding size', 'Max length', 'LSTM parameter', 'Dropout rates', 'Dense nodes', 'Epochs', and 'Batch_size'.
2. You need to collect all the result after running the file.
3. You need to choose the best set of the hyper parameters for the test.

#### Run LSTM_movie_test.ipynb to get the accuracy and the training time for implementing LSTM on the IMDB review dataset
1. You have to collect the 'Wall Time' of training and 'Accuracy' of test after running the file.
 
#### Run LSTM_amazon_test.ipynb to get the accuracy and the training time for implementing LSTM on the Amazon review dataset
1. You have to collect the 'Wall Time' of training and 'Accuracy' of test after running the file.

#### Run LSTM_hotel_test.ipynb to get the accuracy and the training time for implementing LSTM on the Hotel review dataset
1. You have to collect the 'Wall Time' of training and 'Accuracy' of test after running the file.

#### Run LSTM_airline_test.ipynb to get the accuracy and the training time for implementing LSTM on the US airline sentiment dataset
1. You have to collect the 'Wall Time' of training and 'Accuracy' of test after running the file.

#### Run LSTM_twitter_test.ipynb to get the accuracy and the training time for implementing LSTM on the Twitter dataset
1. You have to collect the 'Wall Time' of training and 'Accuracy' of test after running the file.

#### Run LSTM_reddit_test.ipynb to get the accuracy and the training time for implementing LSTM on the Reddit dataset
1. You have to collect the 'Wall Time' of training and 'Accuracy' of test after running the file.

### Run DBN
Open Jupyter Notebook.
Get into the 'DBN' folder.

#### Run DBN_movie_validate.ipynb to get the best set of the hyper parameters for implementing DBN on the IMDB review dataset
1. You have to adjust your own hyper parameters such as 'Hidden_layers_structure', 'Learning_rate_rbm', 'Learning_rate', 'N_epochs_rbm', 'N_iter_backprop', 'Batch_size', and 'Dropout_p'.
2. You need to collect all the result after running the file.
3. You need to choose the best set of the hyper parameters for the test.
 
#### Run DBN_amazon_validate.ipynb to get the best set of the hyper parameters for implementing DBN on the Amazon review dataset
1. You have to adjust your own hyper parameters such as 'Hidden_layers_structure', 'Learning_rate_rbm', 'Learning_rate', 'N_epochs_rbm', 'N_iter_backprop', 'Batch_size', and 'Dropout_p'.
2. You need to collect all the result after running the file.
3. You need to choose the best set of the hyper parameters for the test.

#### Run DBN_hotel_validate.ipynb to get the best set of the hyper parameters for implementing DBN on the Hotel review dataset
1. You have to adjust your own hyper parameters such as 'Hidden_layers_structure', 'Learning_rate_rbm', 'Learning_rate', 'N_epochs_rbm', 'N_iter_backprop', 'Batch_size', and 'Dropout_p'.
2. You need to collect all the result after running the file.
3. You need to choose the best set of the hyper parameters for the test.

#### Run DBN_airline_validate.ipynb to get the best set of the hyper parameters for implementing DBN on the US airline sentiment dataset
1. You have to adjust your own hyper parameters such as 'Hidden_layers_structure', 'Learning_rate_rbm', 'Learning_rate', 'N_epochs_rbm', 'N_iter_backprop', 'Batch_size', and 'Dropout_p'.
2. You need to collect all the result after running the file.
3. You need to choose the best set of the hyper parameters for the test.

#### Run DBN_twitter_validate.ipynb to get the best set of the hyper parameters for implementing DBN on the Twitter dataset
1. You have to adjust your own hyper parameters such as 'Hidden_layers_structure', 'Learning_rate_rbm', 'Learning_rate', 'N_epochs_rbm', 'N_iter_backprop', 'Batch_size', and 'Dropout_p'.
2. You need to collect all the result after running the file.
3. You need to choose the best set of the hyper parameters for the test.

#### Run DBN_reddit_validate.ipynb to get the best set of the hyper parameters for implementing DBN on the Reddit dataset
1. You have to adjust your own hyper parameters such as 'Hidden_layers_structure', 'Learning_rate_rbm', 'Learning_rate', 'N_epochs_rbm', 'N_iter_backprop', 'Batch_size', and 'Dropout_p'.
2. You need to collect all the result after running the file.
3. You need to choose the best set of the hyper parameters for the test.

#### Run DBN_movie_test.ipynb to get the accuracy and the training time for implementing DBN on the IMDB review dataset
1. You have to collect the 'Wall Time' of training and 'Accuracy' of test after running the file.
 
#### Run DBN_amazon_test.ipynb to get the accuracy and the training time for implementing DBN on the Amazon review dataset
1. You have to collect the 'Wall Time' of training and 'Accuracy' of test after running the file.

#### Run DBN_hotel_test.ipynb to get the accuracy and the training time for implementing DBN on the Hotel review dataset
1. You have to collect the 'Wall Time' of training and 'Accuracy' of test after running the file.

#### Run DBN_airline_test.ipynb to get the accuracy and the training time for implementing DBN on the US airline sentiment dataset
1. You have to collect the 'Wall Time' of training and 'Accuracy' of test after running the file.

#### Run DBN_twitter_test.ipynb to get the accuracy and the training time for implementing DBN on the Twitter dataset
1. You have to collect the 'Wall Time' of training and 'Accuracy' of test after running the file.

#### Run DBN_reddit_test.ipynb to get the accuracy and the training time for implementing DBN on the Reddit dataset
1. You have to collect the 'Wall Time' of training and 'Accuracy' of test after running the file.

### Transfer data from Numpy to MATLAB
Open Jupyter Notebook.
Get into the 'numpy_to_matlab' folder.

#### Run movie_validation_to_matlab.ipynb to get the validation set of implementing DELM on the IMDB review dataset
1. You will get a folder named 'npbin_movie' in the 'numpy_to_matlab' folder.
 
#### Run amazon_validation_to_matlab.ipynb to get the validation set of implementing DELM on the Amazon review dataset
1. You will get a folder named 'npbin_amazon' in the 'numpy_to_matlab' folder.

#### Run hotel_validation_to_matlab.ipynb to get the validation set of implementing DELM on the Hotel review dataset
1. You will get a folder named 'npbin_hotel' in the 'numpy_to_matlab' folder.

#### Run airline_validation_to_matlab.ipynb to get the validation set of implementing DELM on the US airline sentiment dataset
1. You will get a folder named 'npbin_airline' in the 'numpy_to_matlab' folder.

#### Run twitter_validation_to_matlab.ipynb to get the validation set of implementing DELM on the Twitter dataset
1. You will get a folder named 'npbin_twitter' in the 'numpy_to_matlab' folder.

#### Run reddit_validation_to_matlab.ipynb to get the validation set of implementing DELM on the Reddit dataset
1. You will get a folder named 'npbin_reddit' in the 'numpy_to_matlab' folder.

#### Run movie_test_to_matlab.ipynb to get the test set of implementing DELM on the IMDB review dataset
1. You will get a folder named 'npbin_test_movie' in the 'numpy_to_matlab' folder.
 
#### Run amazon_test_to_matlab.ipynb to get the test set of implementing DELM on the Amazon review dataset
1. You will get a folder named 'npbin_test_amazon' in the 'numpy_to_matlab' folder.

#### Run hotel_test_to_matlab.ipynb to get the test set of implementing DELM on the Hotel review dataset
1. You will get a folder named 'npbin_test_hotel' in the 'numpy_to_matlab' folder.

#### Run airline_test_to_matlab.ipynb to get the test set of implementing DELM on the US airline sentiment dataset
1. You will get a folder named 'npbin_test_airline' in the 'numpy_to_matlab' folder.

#### Run twitter_test_to_matlab.ipynb to get the test set of implementing DELM on the Twitter dataset
1. You will get a folder named 'npbin_test_twitter' in the 'numpy_to_matlab' folder.

#### Run reddit_test_to_matlab.ipynb to get the test set of implementing DELM on the Reddit dataset
1. You will get a folder named 'npbin_test_reddit' in the 'numpy_to_matlab' folder.

### Run DELM
Open MATLAB R2019a.
Get into the 'DELM' folder.
Open the 'delm.mlx' file.

#### Run delm.mlx to get the best set of the hyper parameters for implementing DELM on the IMDB review dataset
1. You need to copy a file prefixed with testdata from the 'npbin_movie' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 4th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
2. You need to copy a file prefixed with testlabel from the 'npbin_movie' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 5th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
3. You need to copy a file prefixed with traindata from the 'npbin_movie' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 6th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
4. You need to copy a file prefixed with trainlabel from the 'npbin_movie' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 7th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
5. You have to adjust your own hyper parameters such as 'TotalLayers', 'HiddenNeurons', 'C1', 'RhoValue', 'Sigpara', and 'Sigpara1' in the 1st section in the 'delm.mlx' file. Details of the function are in the 'MELM_MNIST25.m' file.
6. You have to run the 1st section in the 'delm.mlx' file.
7. You need to collect all the result after running the file.
8. You need to choose the best set of the hyper parameters for the test.
 
#### Run delm.mlx to get the best set of the hyper parameters for implementing DELM on the Amazon review dataset
1. You need to copy a file prefixed with testdata from the 'npbin_amazon' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 4th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
2. You need to copy a file prefixed with testlabel from the 'npbin_amazon' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 5th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
3. You need to copy a file prefixed with traindata from the 'npbin_amazon' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 6th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
4. You need to copy a file prefixed with trainlabel from the 'npbin_amazon' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 7th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
5. You have to adjust your own hyper parameters such as 'TotalLayers', 'HiddenNeurons', 'C1', 'RhoValue', 'Sigpara', and 'Sigpara1' in the 1st section in the 'delm.mlx' file. Details of the function are in the 'MELM_MNIST25.m' file.
6. You have to run the 1st section in the 'delm.mlx' file.
7. You need to collect all the result after running the file.
8. You need to choose the best set of the hyper parameters for the test.

#### Run delm.mlx to get the best set of the hyper parameters for implementing DELM on the Hotel review dataset
1. You need to copy a file prefixed with testdata from the 'npbin_hotel' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 4th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
2. You need to copy a file prefixed with testlabel from the 'npbin_hotel' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 5th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
3. You need to copy a file prefixed with traindata from the 'npbin_hotel' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 6th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
4. You need to copy a file prefixed with trainlabel from the 'npbin_hotel' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 7th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
5. You have to adjust your own hyper parameters such as 'TotalLayers', 'HiddenNeurons', 'C1', 'RhoValue', 'Sigpara', and 'Sigpara1' in the 1st section in the 'delm.mlx' file. Details of the function are in the 'MELM_MNIST25.m' file.
6. You have to run the 1st section in the 'delm.mlx' file.
7. You need to collect all the result after running the file.
8. You need to choose the best set of the hyper parameters for the test.

#### Run delm.mlx to get the best set of the hyper parameters for implementing DELM on the US airline sentiment dataset
1. You need to copy a file prefixed with testdata from the 'npbin_airline' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 4th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
2. You need to copy a file prefixed with testlabel from the 'npbin_airline' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 5th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
3. You need to copy a file prefixed with traindata from the 'npbin_airline' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 6th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
4. You need to copy a file prefixed with trainlabel from the 'npbin_airline' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 7th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
5. You have to adjust your own hyper parameters such as 'TotalLayers', 'HiddenNeurons', 'C1', 'RhoValue', 'Sigpara', and 'Sigpara1' in the 1st section in the 'delm.mlx' file. Details of the function are in the 'MELM_MNIST25.m' file.
6. You have to run the 1st section in the 'delm.mlx' file.
7. You need to collect all the result after running the file.
8. You need to choose the best set of the hyper parameters for the test.

#### Run delm.mlx to get the best set of the hyper parameters for implementing DELM on the Twitter dataset
1. You need to copy a file prefixed with testdata from the 'npbin_twitter' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 4th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
2. You need to copy a file prefixed with testlabel from the 'npbin_twitter' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 5th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
3. You need to copy a file prefixed with traindata from the 'npbin_twitter' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 6th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
4. You need to copy a file prefixed with trainlabel from the 'npbin_twitter' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 7th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
5. You have to adjust your own hyper parameters such as 'TotalLayers', 'HiddenNeurons', 'C1', 'RhoValue', 'Sigpara', and 'Sigpara1' in the 1st section in the 'delm.mlx' file. Details of the function are in the 'MELM_MNIST25.m' file.
6. You have to run the 1st section in the 'delm.mlx' file.
7. You need to collect all the result after running the file.
8. You need to choose the best set of the hyper parameters for the test.

#### Run delm.mlx to get the best set of the hyper parameters for implementing DELM on the Reddit dataset
1. You need to copy a file prefixed with testdata from the 'npbin_reddit' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 4th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
2. You need to copy a file prefixed with testlabel from the 'npbin_reddit' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 5th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
3. You need to copy a file prefixed with traindata from the 'npbin_reddit' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 6th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
4. You need to copy a file prefixed with trainlabel from the 'npbin_reddit' folder to the 'npbin' folder. Then run the 2nd section in the 'delm.mlx' file. Next run the 7th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
5. You have to adjust your own hyper parameters such as 'TotalLayers', 'HiddenNeurons', 'C1', 'RhoValue', 'Sigpara', and 'Sigpara1' in the 1st section in the 'delm.mlx' file. Details of the function are in the 'MELM_MNIST25.m' file.
6. You have to run the 1st section in the 'delm.mlx' file.
7. You need to collect all the result after running the file.
8. You need to choose the best set of the hyper parameters for the test.

#### Run delm.mlx to get the accuracy and the training time for implementing DELM on the IMDB review dataset
1. You need to copy a file prefixed with testdata from the 'npbin_test_movie' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 4th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
2. You need to copy a file prefixed with testlabel from the 'npbin_test_movie' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 5th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
3. You need to copy a file prefixed with traindata from the 'npbin_test_movie' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 6th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
4. You need to copy a file prefixed with trainlabel from the 'npbin_test_movie' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 7th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
5. You have to adjust your own hyper parameters such as 'TotalLayers', 'HiddenNeurons', 'C1', 'RhoValue', 'Sigpara', and 'Sigpara1' in the 1st section in the 'delm.mlx' file. Details of the function are in the 'MELM_MNIST25.m' file.
6. You have to run the 1st section in the 'delm.mlx' file.
7. You need to collect all the result after running the file.
8. You have to collect the training time and the testing accuracy after running the file.
 
#### Run delm.mlx to get the accuracy and the training time for implementing DELM on the Amazon review dataset
1. You need to copy a file prefixed with testdata from the 'npbin_test_amazon' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 4th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
2. You need to copy a file prefixed with testlabel from the 'npbin_test_amazon' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 5th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
3. You need to copy a file prefixed with traindata from the 'npbin_test_amazon' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 6th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
4. You need to copy a file prefixed with trainlabel from the 'npbin_test_amazon' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 7th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
5. You have to adjust your own hyper parameters such as 'TotalLayers', 'HiddenNeurons', 'C1', 'RhoValue', 'Sigpara', and 'Sigpara1' in the 1st section in the 'delm.mlx' file. Details of the function are in the 'MELM_MNIST25.m' file.
6. You have to run the 1st section in the 'delm.mlx' file.
7. You need to collect all the result after running the file.
8. You have to collect the training time and the testing accuracy after running the file.

#### Run delm.mlx to get the accuracy and the training time for implementing DELM on the Hotel review dataset
1. You need to copy a file prefixed with testdata from the 'npbin_test_hotel' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 4th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
2. You need to copy a file prefixed with testlabel from the 'npbin_test_hotel' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 5th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
3. You need to copy a file prefixed with traindata from the 'npbin_test_hotel' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 6th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
4. You need to copy a file prefixed with trainlabel from the 'npbin_test_hotel' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 7th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
5. You have to adjust your own hyper parameters such as 'TotalLayers', 'HiddenNeurons', 'C1', 'RhoValue', 'Sigpara', and 'Sigpara1' in the 1st section in the 'delm.mlx' file. Details of the function are in the 'MELM_MNIST25.m' file.
6. You have to run the 1st section in the 'delm.mlx' file.
7. You need to collect all the result after running the file.
8. You have to collect the training time and the testing accuracy after running the file.

#### Run delm.mlx to get the accuracy and the training time for implementing DELM on the US airline sentiment dataset
1. You need to copy a file prefixed with testdata from the 'npbin_test_airline' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 4th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
2. You need to copy a file prefixed with testlabel from the 'npbin_test_airline' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 5th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
3. You need to copy a file prefixed with traindata from the 'npbin_test_airline' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 6th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
4. You need to copy a file prefixed with trainlabel from the 'npbin_test_airline' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 7th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
5. You have to adjust your own hyper parameters such as 'TotalLayers', 'HiddenNeurons', 'C1', 'RhoValue', 'Sigpara', and 'Sigpara1' in the 1st section in the 'delm.mlx' file. Details of the function are in the 'MELM_MNIST25.m' file.
6. You have to run the 1st section in the 'delm.mlx' file.
7. You need to collect all the result after running the file.
8. You have to collect the training time and the testing accuracy after running the file.

#### Run delm.mlx to get the accuracy and the training time for implementing DELM on the Twitter dataset
1. You need to copy a file prefixed with testdata from the 'npbin_test_twitter' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 4th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
2. You need to copy a file prefixed with testlabel from the 'npbin_test_twitter' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 5th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
3. You need to copy a file prefixed with traindata from the 'npbin_test_twitter' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 6th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
4. You need to copy a file prefixed with trainlabel from the 'npbin_test_twitter' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 7th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
5. You have to adjust your own hyper parameters such as 'TotalLayers', 'HiddenNeurons', 'C1', 'RhoValue', 'Sigpara', and 'Sigpara1' in the 1st section in the 'delm.mlx' file. Details of the function are in the 'MELM_MNIST25.m' file.
6. You have to run the 1st section in the 'delm.mlx' file.
7. You need to collect all the result after running the file.
8. You have to collect the training time and the testing accuracy after running the file.

#### Run delm.mlx to get the accuracy and the training time for implementing DELM on the Reddit dataset
1. You need to copy a file prefixed with testdata from the 'npbin_test_reddit' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 4th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
2. You need to copy a file prefixed with testlabel from the 'npbin_test_reddit' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 5th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
3. You need to copy a file prefixed with traindata from the 'npbin_test_reddit' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 6th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
4. You need to copy a file prefixed with trainlabel from the 'npbin_test_reddit' folder to the 'npbin' folder. Then run the 3rd section in the 'delm.mlx' file. Next run the 7th section in the 'delm.mlx' file. At last delete the file in the 'npbin' folder.
5. You have to adjust your own hyper parameters such as 'TotalLayers', 'HiddenNeurons', 'C1', 'RhoValue', 'Sigpara', and 'Sigpara1' in the 1st section in the 'delm.mlx' file. Details of the function are in the 'MELM_MNIST25.m' file.
6. You have to run the 1st section in the 'delm.mlx' file.
7. You need to collect all the result after running the file.
8. You have to collect the training time and the testing accuracy after running the file.
