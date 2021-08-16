# Stage_Telecom

 
In DNNs, some specific training points can have a big impact on the model's decision than other training points in making the prediction for test images.
The aim of this project is to tract training points which are most influential to the decision of a denoising neural network, FFDNet.

You can read the details of this project in **LINK**.


Python: 3.8.1, Pytorch: **ver check**, and Implemented on Spyder
Here, I uploaded all the data for our experiments: **LINK**
Here, I uploaded all the code for our experiments:



### Files
-------------------------------------------------------

We conducted experiments on two model: MNIST classifier, FFDNet



About all the code for MNIST, you can find them in mnist_test directory.

#### MNIST (in mnist_test directory)
- datasets: datasets for mnist, we used built-in mnist data set in pytorch, but for our experiments, we needed to manipulate the dataset. I put the codes in this directory.
- model.py: mnist classifier model
- Train.py: train mnist model, outputs trained model, gradient norm vector (for our experiments).
- Test.py: test the trained mnist model.
- m1_Test.py: calculate influence values (gradient norm * activations) for method 1.
- m2_Test.py: calculate influence values for method 2 (from Influence function paper).
- staTest.py: count how many training points are same obtained from different models.
- expTrain.py: train new model with manipulated dataset, outputs trained model with manipulated dataset.
- expTest.py: test the new trained model, outputs losses and accuracy.
- vis_m1_grad.py: visualize the results of method1 (explanatory images on network).
- vis_m1.py: visualize the results of method1 (explanatory images on test images from method1).
- vis_m2.py: visualize the results of method2 (explanatory images on test images from method2).
- vis_staTest_m1.py: visualize the results of method1 from different models.
- vis_staTest_m2.py: visualize the results of method2 from different models.
- utils_if.py: functions for method2.
- utils_option.py: functions for mnist_test.
- Implementation.json: the setting values for mnist test.



From here, I explain the files for FFDNet experiments.

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

 
 
