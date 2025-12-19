# CMDNet: Concrete MAP Detection Network
[![DOI](https://zenodo.org/badge/696866184.svg)](https://doi.org/10.5281/zenodo.8416507)

Original CMDNet implementation for Soft MIMO Detection in TensorFlow 1. Furthermore, competitive MIMO detectors were implemented in TensorFlow 1 and NumPy. It is the source code from the scientific research articles [1, 2].

## Related Repositories

- [`cmdnet-sionna-tf2`](https://github.com/ant-uni-bremen/cmdnet-sionna-tf2): A new version was adapted for TensorFlow 2 and Sionna 0.9.0 including joint soft detection and decoding, being able to reproduce the tf1 simulations.

We created 3 forks to compare to the following original implementations: 
- [`MMNet_for_CMDNet`](https://github.com/ant-uni-bremen/MMNet_for_CMDNet)
- [`LearningToDetect_for_CMDNet`](https://github.com/ant-uni-bremen/LearningToDetect_for_CMDNet)
- [`DeepMIMODetection_for_CMDNet`](https://github.com/ant-uni-bremen/DeepMIMODetection_for_CMDNet)

The original code was downloaded, modified and resimulated for comparison to CMDNet enabling high reproducibility in publication [1, 2]. In all 3 forks, we first simulate the original code and then save the graph including its weights such that it can be loaded into the CMDNet script for evaluation.

## Publications
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “CMDNet: Learning a Probabilistic Relaxation of Discrete Variables for Soft Detection With Low Complexity,” IEEE Trans. Commun., vol. 69, no. 12, pp. 8214–8227, Dec. 2021. https://doi.org/10.1109/TCOMM.2021.3114682
2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Concrete MAP Detection: A Machine Learning Inspired Relaxation,” in 24th International ITG Workshop on Smart Antennas (WSA 2020), vol. 24, Hamburg, Germany, Feb. 2020, pp. 1–5.
3. Matthias Hummert, Dirk Wübben, and Armin Dekorsy, “DeEQ: Deep equalization for large MIMO systems,” in 24th International ITG Workshop on Smart Antennas (WSA 2020), vol. 24, Hamburg, Germany, Feb. 2020, pp. 1–6.

# Requirements & Usage

This code was written for `TensorFlow 1.15` and the dependencies can be found in the Conda environment `conda_env_cmdnet.yml` (see `conda_env_cmdnet_win.yml` for Windows).

Run the script as `python3 cmdnet_mimo_detection.py`, to reproduce the results of the articles. To do so, set the parameters in the file `sim_settings.yaml` to the values in the articles.

There are 5 template setting files:
1. Default training
2. Joint soft detection + decoding
3. LLR (Log-Likelihood Ratio) plot
4. Online learning CMDNet
5. Online learning DNN
6. DeEQ MIMO settings for comparison to [2, 3]


# Acknowledgements

This work was partly funded by the German Ministry of Education and Research (BMBF) under grant 16KIS1028 (MOMENTUM).

# License and Referencing

This program is licensed under the GPLv3 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

# Abstract of the Publications

1. Following the great success of Machine Learning (ML), especially Deep Neural Networks (DNNs), in many research domains in 2010s, several ML-based approaches were proposed for detection in large inverse linear problems, e.g., massive MIMO systems. The main motivation behind is that the complexity of Maximum A-Posteriori (MAP) detection grows exponentially with system dimensions. Instead of using DNNs, essentially being a black-box, we take a slightly different approach and introduce a probabilistic Continuous relaxation of disCrete variables to MAP detection. Enabling close approximation and continuous optimization, we derive an iterative detection algorithm: Concrete MAP Detection (CMD). Furthermore, extending CMD by the idea of deep unfolding into CMDNet, we allow for (online) optimization of a small number of parameters to different working points while limiting complexity. In contrast to recent DNN-based approaches, we select the optimization criterion and output of CMDNet based on information theory and are thus able to learn approximate probabilities of the individual optimal detector. This is crucial for soft decoding in today’s communication systems. Numerical simulation results in MIMO systems reveal CMDNet to feature a promising accuracy complexity trade-off compared to State of the Art. Notably, we demonstrate CMDNet’s soft outputs to be reliable for decoders.

2. Motivated by large linear inverse problems where the complexity of the Maximum A-Posteriori (MAP) detector grows exponentially with system dimensions, e.g., large MIMO, we introduce a method to relax a discrete MAP problem into a continuous one. The relaxation is inspired by recent ML research and offers many favorable properties reflecting its quality. Hereby, we derive an iterative detection algorithm based on gradient descent optimization: Concrete MAP Detection (CMD). We show numerical results of application in large MIMO systems that demonstrate superior performance w.r.t. all considered State of the Art approaches.

3. Multiple Input Multiple Output (MIMO) and massive MIMO (mMIMO) are key-enabling technologies for 4G and 5G communications systems. mMIMO uses a high number of antennas, where the number of antennas at the base station exceeds in general the number of antennas in the mobiles. For uncorrelated channels, linear equalizers already achieve promising performance in the uplink due to the channel hardening effect. In contrast, we will focus on large symmetrical MIMO systems in this paper, where many antennas are employed at the transmitter and the receiver side resulting in a more challenging task for the receiver. Traditionally, receiver algorithms have been derived based on models for the communications system. Recently, machine learning approaches have been proposed where the design is data driven. In order to overcome the drawbacks of model-based and pure data driven approaches, hybrid approaches combining the benefits of both worlds have emerged. In this paper, we present the novel hybrid approach entitled Deep Equalization (DeEQ) based on model knowledge and a neural network like structure. As demonstrated by simulation results, this novel approach achieves very good performance with the advantage of only a very low error floor.