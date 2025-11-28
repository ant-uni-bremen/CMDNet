# DeepMIMODetection for CMDNet

The original DetNet code from [2] was downloaded, modified and resimulated for comparison to CMDNet enabling high reproducibility in publication [1]:
1. Edgar Beck, Carsten Bockelmann, and Armin Dekorsy, “CMDNet: Learning a Probabilistic Relaxation of Discrete Variables for Soft Detection With Low Complexity,” IEEE Trans. Commun., vol. 69, no. 12, pp. 8214–8227, Dec. 2021. https://doi.org/10.1109/TCOMM.2021.3114682
2. N. Samuel, T. Diskin, and A. Wiesel, “Deep MIMO Detection,” in 18th IEEE International Workshop on Signal Processing Advances in Wireless Communications (SPAWC 2017), Sapporo, Japan, Jul. 2017, pp. 1–5. https://doi.org/10.1109/SPAWC.2017.8227772

The main files are ´DetNet_CMDNet.py´ and ´load_DetNet_CMDNet.py´ besides the original main files.


## DeepMIMODetection (Original)

This repository holds 2 files.

The first one is the FullyConnected architecture used to detect over a fixed channel (in this specific case we detect over a 0.6-Toeplitz channel where K=20 N=30. CSV file with the channel also found in this repo).

The second one is the DetNet architecture as described in the paper "Deep MIMO Detection" presented at SPAWC 2017. Detection is over I.I.D gaussian random channels.

For any further questions regarding the code contact at neev.samuel@gmail.com
