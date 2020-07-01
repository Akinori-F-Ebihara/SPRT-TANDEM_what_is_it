# SPRT-TANDEM: what is it?
A casual introduction to the SPRT-TANDEM algorithm. (under construction)

## Introduction
SPRT-TANDEM is a sequential density ratio estimation algorithm originally proposed in the paper, "Deep Neural Networks for the Sequential Probability Ratio Test on Non-i.i.d. Data Series". The SPRT-TANDEM sequentially estimates log-likelihood ratios of two hypotheses, or classes, for fast and accurate sequential data classification. 

The original paper [1] can be found here:  
https://arxiv.org/abs/2006.05587


The tensorflow implementation of the SPRT-TANDEM can be found here:  
https://github.com/TaikiMiyagawa/SPRT-TANDEM


While the technical details are left to the paper, we provide a casual introduction to our algorithm below.

## References
[1] A. F. Ebihara, T. Miyagawa, K. Sakurai, and H. Imaoka. Deep neural networks for the sequential probability ratiotest on non-i.i.d. data series, __arXiv__, 2020  
[2] S. Kira, T. Yang, and M. N. Shadlen. A neural implementation of wald’s sequential probability rato test. ___Neuron___, 85(4):861–873, Feb. 2015.  
[3] A. Tartakovsky,  I. Nikiforov,  and M. Basseville.Sequential Analysis: Hypothesis Testing and ChangepointDetection. Chapman & Hall/CRC, 1st edition, 2014.  
[4] A. Wald. Sequential tests of statistical hypotheses. ___Ann. Math. Statist.___, 16(2):117–186, 06 1945.
[5] A. Wald.Sequential Analysis. John Wiley and Sons, 1st edition, 1947.
