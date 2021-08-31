# Stage_Telecom

 
In DNNs, some specific training points can have a big impact on the model's decision on a certain test image than other training points.   
The aim of this project is to tract training points which are most influential to the decision of a denoising neural network, FFDNet.   
   
You can read the details of this project in Stage_report.pdf   
<br />
   
Python: 3.8.1, Pytorch: 1.8.1, and Implemented on Spyder   
Here, I uploaded all the data for our experiments: [Data](https://drive.google.com/drive/folders/1yK_4DgJzb4Ify3Tp7nRX3B2mD8qlQ2iI?usp=sharing)   
Here, I uploaded all the code for our experiments:   

You need to put the sub-directories in "ffdnet" directory (from data) next to the FFDNet codes and put the sub-directories in "mnist" directory (from data) next to the mnist codes in "mnist_test" directory (from github).  
*caution: In the case of mnist, you'll see two directory having same name (named datasets), please put the sub-directories in "mnist/datasets" directory (from data) into "mnist_test/datasets" directory (from github).

<br />
   
   
### Files  
-------------------------------------------------------
  
We conducted experiments on two model: MNIST classifier, FFDNet  
About all the code for MNIST, you can find them in "mnist_test" directory.
  
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

<br />
Here, I explain the parameters in **"implementation.json"**.  
These parameters are all for the FFDNet. The parameters for our test, you can find them in the bottom.
  
- seed: seed number
- use_gpu: whether to use gpu or not
- is_gray: whether to implement with 'gray' FFDNet or 'rgb' FFDNet. I always set this to false (rgb) when I tested.
- is_clip: whether to clip the denoised image. I set this to true when I tested.
<br />

**dataset**
- phase: this is used when I make the dataset. (see details in "DatasetFFD.py" file)
- patch_size: patch_size. I always set 64 when I tested.
- num_workers: num_workers value for DataLoader
- is_gray: whether gray image or rgb image. I always set this to false (rgb) when I tested.
- base_path: base_path for dataset
- sigma: sigma value for noised image.
- sigma_test: sigma_test value
  
**train**
- logger_: logger name and path
- loss_fn: 'l2' or 'l1'. loss function
- reduction: 'mean'. used in loss function
- optimizer: 'adam' or 'sgd'. I set this to 'adam' when I tested.
- learning_rate: learning_rate. I set this to 1e-3 when I tested.
- epoch: epoch. I set this to 20 when I tested.
- batch_size: batch_size. I set this to 8 when I tested.
- val_epoch: epoch num when I want to check validation set accuracy.
- train_checkpoints: checkpoints for trained model.
  
**test**
- noise_level_img: noise level on image
- noise_level_model: noise level on model
- border: when you compute PSNR or SSIM, if you want to exclude pixels on border, you can set this value. I set this to 'false' when I tested.
  
<br />
From here, the parameters are for our tests.
  
**method1**
- mode: '1000' or 'normalized'. "normalized" means when I compute influence values, I want to use normalized gradient norm. "1000" means I want to use just the gradient norm without any modification. (I named it '1000' because I used 1,000 training images for our test to make it faster.) (you can see the process of this modification (normalizing or not) in "m1_test.py" file.
- remove_out: whether to remove the outliers or not. ('false' or numbers you want to set as outlier limit.) (you can see the code (FindOutlier) at the bottom in utils_option.py  
- epoch: epoch. I set this to 20 when I tested.
- batch_size: batch size. I set this to 8 when I tested.
- point: when I compute the influence values, I need to choose a point in test image. (See details in "m1_Test.py" file)  
  baboon: [50, 50], [50, 100], [50, 150], [50, 200]  
  barbara: [250, 50], [250, 100], [250, 150], [250, 200]
- layer: when I compute the influence values, I need to choose a layer in the network. (See details in "m1_Test.py" file)
- vis_num: 1(baboon) or 5(barbara). for which test image you want to track the explanatory points.
  
**method2**
- if_dir: directory where I want to save the influence values.
- testset: test set name I want to test. I set this 'set14'.
- use_gpu: whether to use gpu or not. I always set this 'true'.
- damp: "true" or "false". From paper, it used damp for stability.
- stochastic: "true" or "false". when I choose training points for computing the influence values, when I set 'true', it choose 1 point, but when I set this to 'false', it choose the amount of batch size (8). I set this to 'false' for the speed. (but in paper, it chose the point stochastically)
- epoch: epoch. I set this to 20 when I tested.
- batch_size: batch size I set this to 8 when I tested.
- test_sample_num: test sample number. I tested two images (baboon, barbara), so I set this to 2.
- recursion_nums: recursion numbers (see details in influence function paper). I set this 5 when I tested.
- training_points: training points (see details in influence function paper). I set this to 1000 when I tested.
