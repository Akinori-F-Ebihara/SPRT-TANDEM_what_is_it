# SPRT-TANDEM: what is it?
A casual introduction to the SPRT-TANDEM algorithm. (under construction)

## Introduction
SPRT-TANDEM is a sequential density ratio estimation algorithm originally proposed in the paper, "Deep Neural Networks for the Sequential Probability Ratio Test on Non-i.i.d. Data Series". The SPRT-TANDEM sequentially estimates log-likelihood ratios of two hypotheses, or classes, for fast and accurate sequential data classification. 

The original paper [1] can be found here:  
https://arxiv.org/abs/2006.05587

The tensorflow implementation of the SPRT-TANDEM can be found here:  
https://github.com/TaikiMiyagawa/SPRT-TANDEM


While the technical details are left to the paper, we provide a casual introduction to our algorithm below.

## Problem setting
Imagine you have a sequential data, 
( x^(1), x^(2), ...),
where x^(n) can be a data sample, such as a video frame, an audio signal, a neural firing rate, etc. The sequential data has an associated label $y$ that indicates a class to which the sequential data belong. Your task is to correctly estimate the class label $y$, with a minimal number of data samples (say, $x_1, ... x_k$ samples where $k < n$). Generally speaking, there is a trade-off between speed (i.e., small data samples) and accuracy (i.e., a correct estimate of the label): using fewer samples tend to increase the misclassification rate, while high accurate classification requires more data samples. It is a non-trivial problem.

## Sequential Probability Ratio Test
One algorithm that provides a solution to the above tradeoff problem is the Sequential Probability Ratio Test, or SPRT, which was originally invented by Abraham Wald [4, 5]. The SPRT calculates the log-likelihood ratio (LLR) of two competing hypotheses and updates the LLR every time a new sample is acquired until the LLR reaches one of the two thresholds for alternative hypotheses.

(figure:SPRT)

As the figure above shows, for data that is easy to classify, the SPRT outputs an answer taking a few samples, whereas, for difficult data, the SPRT takes in numerous samples in order to make a ``careful'' decision. Importantly, Wald and his colleagues proved that when sequential data are sampled from independently and identically distributed (i.i.d.) data, SPRT can minimize the required number of samples to achieve the desired upper-bounds of false positive and false negative rates comparably to the
Neyman-Pearson test, known as the most powerful likelihood test [3, 5]

## Example 1: coin flipping
Let's start with a toy example to get the hang of the SPRT.\\
You have two coins, but one of them is a skewed coin that has uneven probabilities of generating head or tail when it is flipped:

coin A: 1/2head, 1/2tail
coin B: 1/3head, 2/3tail

Coins are unlabeled, and you do not know which one is the coin A. Now, you want to experiment with the two coins. Flipping each of them six times yields the following results.

The first coin:
$X_1 = {\mathrm{head, head, tail, head, tail, tail}}$\\

The second coin:
$X_2 = {\mathrm{tail, head, tail, head, tail, tail}}$

you have two hypotheses: \\ $H_0$: It is the coin A. \\ $H_1$: It is the coin B.
In this toy example, you can calculate the exact log-likelihood ratio for $X_1$ and $X_2$, because you know the probabilities of being head or tail:

(likelihood calculation)

Note that flipping trials are independent. Thus, the first coin is likely to be coin A, while the second coin is coin B.


## Example 2: face spoofing detection
\subsection{Example 2: face spoofing detection}
Next, let's consider a more realistic scenario: face spoofing detection. Face spoofing detection is one of the biometrics task classifying a facial image into a live face class, or a spoof face class (e.g., a facial photo, a face displayed on a screen, a face mask).\\

In this example, you are presented with a series of facial image to choose one of the two hypotheses,\\

$H_0$: It is a live face. \\
$H_1$: It is a spoof face. \\ 

Now let's see an example. 

(image)
(image)

Here, you are confronting with two problems executing the SPRT. First, unlike the coin-flipping example, you do not know the generating probability of the given data. Second, the video frames are highly correlated, and the assumption of the original SPRT no longer holds. These two problems hamper calculating the likelihood ratio.

## SPRT-TANDEM for the likelihood estimation
So what should we do? Here comes the SPRT-TANDEM algorithm. We use two kinds of density ratio estimation algorithms, ratio matching approach, and probabilistic classification approach, to let a deep neural network estimate the likelihood ratio. To control a correlation length that is considered, we propose the TANDEM formula:

(TANDEM formula).

Our proposed neural network is trained to explicitly calculate the TANDEM formula to provide the sequential likelihood ratio estimation. Below is the estimated likelihood trajectories of example 2.

(figure: SPRT-TANDEM)


# SPRT as an algorithm of the brain
The SPRT algorithm makes an early decision for an easy data series, while it takes time to make a decision on a difficult data. This is quite in line with our daily mental process - the more difficult a problem is, the longer time we require for decision making. Indeed, the SPRT seems to be the best algorithm explaining neural activities in the primate brain. Kira et al. [2] found that neurons in the part of the primate brain called the lateral intraparietal cortex (LIP) showed neural activities reminiscent of the SPRT; when a monkey sequentially collecs random pieces of evidence to make a binary choice, LIP neurons show activities proportional to the LLR. Note that the presented stimuli are distributed i.i.d.; thus, it remains an open question if the brain uses the SPRT-TANDEM for correlated data or uses some other algorithm. 

## References
[1] A. F. Ebihara, T. Miyagawa, K. Sakurai, and H. Imaoka. Deep neural networks for the sequential probability ratiotest on non-i.i.d. data series, __arXiv__, 2020

[2] S. Kira, T. Yang, and M. N. Shadlen. A neural implementation of wald’s sequential probability rato test. ___Neuron___, 85(4):861–873, Feb. 2015.

[3] A. Tartakovsky,  I. Nikiforov,  and M. Basseville.Sequential Analysis: Hypothesis Testing and ChangepointDetection. Chapman & Hall/CRC, 1st edition, 2014.

[4] A. Wald. Sequential tests of statistical hypotheses. ___Ann. Math. Statist.___, 16(2):117–186, 06 1945.

[5] A. Wald.Sequential Analysis. John Wiley and Sons, 1st edition, 1947.

