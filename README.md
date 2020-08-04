# J-MoDL
Joint model-based deep learning for parallel imaging.

Reference article: J-MoDL: Joint Model-Based Deep Learning for Optimized Sampling and Reconstruction by H.K Aggarwal and M. Jacob in IEEE Journal of Selected Topics in Signal Processing, (2020). 

arXiv link: https://arxiv.org/abs/1911.02945

IEEEXplore: https://ieeexplore.ieee.org/document/9122388

#### How to run the codes.

We have tested the codes in Tensorflow 1.15 but it can run in other versions as well. 
This git repository includes a test image with coil sensitivities in the file `tstdata_jmodl.npz`. The testing script `tst.py` will use this image by default and does not require full data download for the testing purpose.

We have also released a subset of the parallel imaging dataset used in this paper. This data is required in the training code `trn.py`.  You can download the dataset from the below link:

 **Dataset Download** :  https://drive.google.com/file/d/1GLqs2A5YpRN8RdDJgdhrspL3zjlG0Qha/view?usp=sharing

You can run the test code without making any change as:

`$python tst.py`. It should give following output.
#### Output of Joint learning the sampling pattern and network parameters:
![alt text](https://github.com/hkaggarwal/J-MoDL/blob/master/output.jpeg)

#### Proposed continuous optimization scheme in the 2D case
In this paper, we have proposed how to learn the sampling mask together with the reconstruction network in the parallel imaging settings.
As shown in this diagram we train for the real valued sampling locations rather than the entire sampling mask.

![alt text](https://github.com/hkaggarwal/J-MoDL/blob/master/continuous_optimization.jpg)


#### Overall J-MoDL architecture
This is the structure of the overall network. Please refer to the paper for details on notations.

`A_theta` is the acquisition operator having trainable sampling parameter `theta`.

`D_phi` is the CNN network with trainable parameters `phi`.

`Q_theta` is the data-consistency block that allows to use imaging physics.

![alt text](https://github.com/hkaggarwal/J-MoDL/blob/master/j-modl_architecture.jpg)

#### Contact
The code is provided to support reproducible research. If the code is giving syntax error in your particular python configuration or some files are missing then you may open an issue or directly email me at jnu.hemant@gmail.com


