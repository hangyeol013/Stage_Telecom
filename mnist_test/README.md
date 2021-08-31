# Stage_Telecom

 
In DNNs, some specific training points can have a big impact on the model's decision on a certain test image than other training points.   
The aim of this project is to tract training points which are most influential to the decision of a denoising neural network, FFDNet.   
   
You can read the details of this project in Stage_report.pdf   
   
   
Python: 3.8.1, Pytorch: **ver check**, and Implemented on Spyder   
Here, I uploaded all the data for our experiments: [Data](https://drive.google.com/drive/folders/1yK_4DgJzb4Ify3Tp7nRX3B2mD8qlQ2iI?usp=sharing)   
Here, I uploaded all the code for our experiments:   
   
   
   
### Files  
-------------------------------------------------------
  
We conducted experiments on two model: MNIST classifier, FFDNet  
  
  
  
Here, I explain all the code in mnist_test directory
  
#### MNIST  
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
<br />
<br />

#### Here I explain the parameters in <<Implementation.json>>
- seed: seed number (I set this seed to '0', when I train staTest0 model and when I do the explainability test.
- use_gpu: whether we use gpu or not. (I always set this to 'true').
- epoch: number of epoch (I always set this to 20).
- batch_size: batch size (I always set this to 8).
- logger_ : name and path of logger (It's just the name or path of logger, I didn't change this).
- vis_num: when we run "vis_" code, it means how many explanatory points do you want to see. (ex. If you set this to 7, you can see 7 explanatory images.)
- method: 'method1' or 'method2'. when I run "expTrain.py" or "expTest.py" code, I set this value.
- newData: 'algo' or 'random' or 'remove_algo' or remove_random', when I run "expTrain.py" or "expTest.py" code, I set this value.
  
  **Method1**  
- mode: "staTest0" or "staTest1" or "staTest2" or "staTest3" or "staTest4" or "staTest5". when I do the stability test, I changed this name.


  **Method2** (for the parameters in utils_if.py)  
- damp: "true" or "false". From paper, it used damp for stability.
- stochastic: "true" or "false". When I choose training points for compute the influence values, when I set 'true', it choose 1 point, when I set 'false', it choose the amount of batch size (8). I set this to 'false' for the speed. (but in paper, it chose the point stochastically)
- mode: "staTest0" or "staTest1" or "staTest2" or "staTest3" or "staTest4" or "staTest5". when I do the stability test, I changed this name.
- test_sample_num: for how many test images do you want to get the explanatory images. (I set this to 10, but when it takes pretty long time, so sometimes I set this value to 2 or 3 for testing this code)
- recursion_nums: recursion numbers (see details in Influence function paper). I set this to 5.
- training_points: training points (see details in Influence function paper). I set this to 1000.
- if_dir: directory where the influence values are saved.
  
 
