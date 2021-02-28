# Reinforcement Learning for AD diagnosis

## Reference 
#### 1. Yoon, Jinsung, James Jordon, and Mihaela van der Schaar. "INVASE: Instance-wise variable selection using neural networks." International Conference on Learning Representations. 2018.   

#### 2. Lee, Juho, et al. "Set transformer: A framework for attention-based permutation-invariant neural networks." International Conference on Machine Learning. PMLR, 2019.   


#### 3. Elsayed, Gamaleldin, et al. "Revisiting spatial invariance with low-rank local connectivity." International Conference on Machine Learning. PMLR, 2020.   


## Overall framework
1. The captured local features and subtle changes are important in dignosis of AD.
2. However, only sparse features are releated to the brain disease, such as Alzheimer's disease.
3. So, local feature selection has been done before considering their dependency.
4. The dependency could be reflected by considering the selected local feautre as a set. 
5. By employing "set transformer", I tried to capture the dependency.
6. Considering that all of preprocessed MR images are aligned into a template, "LRLC" method has been used in selection network.
<p align="center"><img src="./img/fig_1.png" width="100%" height="100%" title="Alzheimer's disease" alt="Framework"></img></p>
