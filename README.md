# SweiNet
**SweiNet is an uncertainty-quantifying shear wave speed (SWS) estimator for ultrasound shear wave elasticity (SWE) imaging.**

<img src="./images/fig_Architecture.png">

SweiNet takes as input a 2D space-by-time array of tracked particle motion.
It outputs the estimated SWS and estimated uncertainty, both in units of meters per second.

SweiNet was originally trained on a large dataset of *in vivo* cervix SWE acquisitions.
The predicted uncertainty is well-calibrated to these data.
However, with a few pre-processing steps, SweiNet can easily be applied to other datasets.
See the example notebook: `example.ipynb`.
