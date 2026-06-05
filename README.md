# kurDISTANCE

A Python package written in C for computing time-series distance measures.

## Features

- **Elastic Measures:** Dynamic Time Warping (DTW), LCSS, EDR, ERP, MSM, TWED, SWALE, Best Path Extraction.
- **Lock-step Measures:** 
	- **Minkowski:** Euclidean, Manhattan, Minkowski, Chebyshev,
	- **L1:** Sorensen, Gower, Soergel, Kulczynski, Canberra, Lorentzian,
	- **Intersection:** Intersection, Wave Hedges, Czekanowski, Motyka, Tanimoto,
	- **Inner Product:** Inner Product, Harmonic Mean, Kumar-Hassebrook, Jaccard, Cosine, Dice,
	- **Square Chord:** Fidelity, Bhattacharyya, Squared-chord, Hellinger, Matusita,
	- **Squared L2:** Squared Euclidean, Clark, Neyman Chi-Square, Pearson Chi-Square, Squared Chi-Square, Divergence, Additive Symmetric Chi-Square, Probabilistic Symmetric Chi-Square,
	- **Entropy:** Kullback-Leibler, Jeffreys, K Divergence, Topsoe, Jensen Shannon, Jensen Difference,
	- **Vicissitude:** Vicis-Wave Hedges, Emanon 2, Emanon 3, Emanon 4, Max-Symmetric Chi-Square, Min-Symmetric Chi-Square.
- **Trend-Based Measures:** Trend-Based Euklidean Distance, Derivative-Based Euklidean Distance.

## Installation

To build and install the package locally from source, clone the repository and run:

```bash
git clone [https://github.com/MateuszZacharecki/kurDISTANCE.git](https://github.com/MateuszZacharecki/kurDISTANCE.git)
cd kurDISTANCE
pip install -e .