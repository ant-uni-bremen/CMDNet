# CMDNet: Concrete MAP Detection Network
[![DOI](https://zenodo.org/badge/696866184.svg)](https://zenodo.org/badge/latestdoi/696866184)

Original CMDNet implementation for Soft MIMO Detection in Tensorflow 1. Also competitive MIMO detectors were implemented in Tensorflow 1 and Numpy.

Source code from scientific research articles [1, 2]:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “CMDNet: Learning a Probabilistic Relaxation of Discrete Variables for Soft Detection With Low Complexity,” IEEE Trans. Commun., vol. 69, no. 12, pp. 8214–8227, Dec. 2021. https://doi.org/10.1109/TCOMM.2021.3114682
2. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “Concrete MAP Detection: A Machine Learning Inspired Relaxation,” in 24th International ITG Workshop on Smart Antennas (WSA 2020), vol. 24, Hamburg, Germany, Feb. 2020, pp. 1–5.

# Requirements & Usage

This code was tested with TensorFlow 1.15 and cannot be used with TensorFlow 2. A new version was adapted for TensorFlow 2 with joint soft detection and decoding not. It does not work with joint decoding as in this version properly: Numerical experiments show that joint CMDNet and decoder performance decreases significantly with float32 instead of float64 computation accuracy.

Run the script as python3 MIMO_detection.py, to reproduce the results of the articles. To do so, set the parameters in the file sim_settings.yaml to the values in the articles. There are four template setting files with default training settings, joint decoding settings, LLR (Log-Likelihood Ratio) plot settings, and online learning settings.

# Acknowledgements

This work was partly funded by the German Ministry of Education and Research (BMBF) under grant 16KIS1028 (MOMENTUM).

# License and Referencing

This program is licensed under the GPLv3 license. If you in any way use this code for research that results in publications, please cite our original article listed above.

# Abstract of the articles

1. Following the great success of Machine Learning (ML), especially Deep Neural Networks (DNNs), in many research domains in 2010s, several ML-based approaches were proposed for detection in large inverse linear problems, e.g., massive MIMO systems. The main motivation behind is that the complexity of Maximum A-Posteriori (MAP) detection grows exponentially with system dimensions. Instead of using DNNs, essentially being a black-box, we take a slightly different approach and introduce a probabilistic Continuous relaxation of disCrete variables to MAP detection. Enabling close approximation and continuous optimization, we derive an iterative detection algorithm: Concrete MAP Detection (CMD). Furthermore, extending CMD by the idea of deep unfolding into CMDNet, we allow for (online) optimization of a small number of parameters to different working points while limiting complexity. In contrast to recent DNN-based approaches, we select the optimization criterion and output of CMDNet based on information theory and are thus able to learn approximate probabilities of the individual optimal detector. This is crucial for soft decoding in today’s communication systems. Numerical simulation results in MIMO systems reveal CMDNet to feature a promising accuracy complexity trade-off compared to State of the Art. Notably, we demonstrate CMDNet’s soft outputs to be reliable for decoders.

2. Motivated by large linear inverse problems where the complexity of the Maximum A-Posteriori (MAP) detector grows exponentially with system dimensions, e.g., large MIMO, we introduce a method to relax a discrete MAP problem into a continuous one. The relaxation is inspired by recent ML research and offers many favorable properties reflecting its quality. Hereby, we derive an iterative detection algorithm based on gradient descent optimization: Concrete MAP Detection (CMD). We show numerical results of application in large MIMO systems that demonstrate superior performance w.r.t. all considered State of the Art approaches.
