# No MCMC for me: Amortized sampling for fast and stable training of energy-based models
Code for the paper:

> Will Grathowhl*, Jacob Kelly*, Milad Hashemi, Mohammad Norouzi, Kevin Swersky, David Duvenaud. "No MCMC for me: Amortized sampling for fast and stable training of energy-based models" _arXiv preprint_ (2020).
> [[arxiv: TODO]](https://arxiv.org/abs/2007.04504) [[bibtex: TODO]](#bibtex)

\*Equal Contribution

<p align="center">
<img align="middle" src="./assets/fig1.png" width="500" />
</p>

Code for implementing **V**ariational **E**ntropy **R**egularized **A**pproximate maximum likelihood (VERA). Contains scripts for training VERA and using VERA for [JEM](https://github.com/wgrathwohl/JEM) training. Code is also available for training semi-supervised models on tabular data, mode counting experiments, and tractable likelihood models.

For more info on me and my work please checkout my [website](http://www.cs.toronto.edu/~wgrathwohl/), [twitter](https://twitter.com/wgrathwohl), or [Google Scholar](https://scholar.google.ca/citations?user=ZbClz98AAAAJ&hl=en). 

Many thanks to my amazing co-authors: [Jacob Kelly](https://jacobjinkelly.github.io/), [Milad Hashemi](https://research.google/people/MiladHashemi/), [Mohammad Norouzi](https://norouzi.github.io/), [Kevin Swersky](http://www.cs.toronto.edu/~kswersky/), [David Duvenaud](http://www.cs.toronto.edu/~duvenaud/).

## Requirements

```
pytorch==1.5.1
torchvision==0.6.1
numpy
scikit-learn
matplotlib
seaborn
tqdm
```

## Usage

### Training

### Evaluation

### Mode counting

## Data

Tabular data for semi-supervised classification must be downloaded manually and placed in `datasets/`.

### HEPMASS

Download `1000_train.csv.gz` and `1000_test.csv.gz` from [here](http://archive.ics.uci.edu/ml/datasets/HEPMASS). Unzip each of these files and place in `datasets/HEPMASS/`.

### HUMAN

Download `UCI HAR Dataset.zip` from [here](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones). Unzip. Rename the resulting folder to `HUMAN/` and place this folder in `datasets/`.

### CROP

Download `data.zip` from [here](https://archive.ics.uci.edu/ml/datasets/Crop+mapping+using+fused+optical-radar+data+set). Unzip. Place the resulting file in `datasets/CROP/`.

## BibTeX
