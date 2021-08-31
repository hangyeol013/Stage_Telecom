# Stage_Telecom

 
In DNNs, some specific training points can have a big impact on the model's decision on a certain test image than other training points.   
The aim of this project is to tract training points which are most influential to the decision of a denoising neural network, FFDNet.   
   
You can read the details of this project in Stage_report.pdf   
<br />
   
Python: 3.8.1, Pytorch: 1.8.1, and Implemented on Spyder   
Here, I uploaded all the data for our experiments: [Data](https://drive.google.com/drive/folders/1yK_4DgJzb4Ify3Tp7nRX3B2mD8qlQ2iI?usp=sharing)   
Here, I uploaded all the code for our experiments:   

You need to put the sub-directories in "ffdnet" directory next to the FFDNet codes and put the sub-directories in "mnist" directory next to the mnist codes in "mnist_test" directory.  
*caution: In the case of mnist, you'll see two identical directory (named datasets), please put the sub-directories in "mnist/datasets" directory into "mnist_test/datasets" directory.
<br />
   
   
### Files  
-------------------------------------------------------
  
We conducted experiments on two model: MNIST classifier, FFDNet  
About all the code for MNIST, you can find them in mnist_test directory.
  
<br />
Here, I explain the files for FFDNet experiments and implementation.json file.  

#### FFDNet  
- model_zoo: directory containing the trained_model.
- results: directory containing the results of FFDNet.
- model.py: FFDNet model
- DatasetFFDNet.py: datasets for FFDNet.
- FFDNet_Train.py: train FFDNet, outputs trained model.
- FFDNet_Test.py: test the trained FFDNet model, outputs the output images.
- FFDNet_Plot.py: plot the accuracy for training set and validation set.
- m1_Train.py: Train FFDNet, outputs trained model, gradient norm tensor (for our experiments).
- m1_Test.py: calculate the influence values (with gradient norm and activations) for method 1.
- m1_PatchImg.py: track the training image from cropped training patch.
- m1_vis.py: visualize the results of method 1 (explanatory images on test images from method 1).
- m2_Test.py: calculate the influence values for method 2 (from Influence function paper).
- m2_vis.py: visualize the results of method 2 (explanatory images on test images from method 2).
- utils_if.py: functions for method 2.
- utils_option.py: functions for FFDNet and experiments.
- utils_network.py: functions for the FFDNet network.
- Inplementation.json: the setting values for FFDNet, you can change all the values used in FFDNet and experiments here.

 
 
