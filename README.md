# No MCMC for me: Amortized sampling for fast and stable training of energy-based models
Code for the paper:

> Will Grathowhl*, Jacob Kelly*, Milad Hashemi, Mohammad Norouzi, Kevin Swersky, David Duvenaud. "No MCMC for me: Amortized sampling for fast and stable training of energy-based models" _arXiv preprint_ (2020).
> [[arxiv]](https://arxiv.org/abs/2010.04230) [[bibtex]](#bibtex)

\*Equal Contribution

<p align="center">
<img align="middle" src="./assets/fig1.png" width="500" />
</p>

Code for implementing **V**ariational **E**ntropy **R**egularized **A**pproximate maximum likelihood (VERA). Contains scripts for training VERA and using VERA for [JEM](https://github.com/wgrathwohl/JEM) training. Code is also available for training semi-supervised models on tabular data, mode counting experiments, and tractable likelihood models experiments.

For more info on me and my work please checkout my [website](http://www.cs.toronto.edu/~wgrathwohl/), [twitter](https://twitter.com/wgrathwohl), or [Google Scholar](https://scholar.google.ca/citations?user=ZbClz98AAAAJ&hl=en). 

Many thanks to my amazing co-authors: [Jacob Kelly](https://jacobjinkelly.github.io/), [Milad Hashemi](https://research.google/people/MiladHashemi/), [Mohammad Norouzi](https://norouzi.github.io/), [Kevin Swersky](http://www.cs.toronto.edu/~kswersky/), [David Duvenaud](http://www.cs.toronto.edu/~duvenaud/).

## Requirements

```markdown
pytorch==1.5.1
torchvision==0.6.1
numpy
scikit-learn
matplotlib
seaborn
tqdm
```

## Usage

### Hyperparameters

A brief explanation of hyperparameters that can be set from flags and their names in the paper. 
- `--clf_weight` Classification weight (`\alpha`)
- `--pg_control` Gradient norm penalty (`\gamma`)
- `--ent_weight` Entropy regularization weight (`\lambda`)
- `--clf_ent_weight` Classification entropy (`\beta`)

### Training

An explanation of flags for different modes of training. Without any of these flags, an unsupervised VERA model will be trained.

- `--clf_only` For training a classifier on its own, i.e. without an EBM as in JEM.
- `--jem` Do JEM training.
- `--labels_per_class` If this is greater than zero, use this many labels per class for semi-supervised learning. If zero (default), do full-label training.

To train a CIFAR10/CIFAR100 JEM model as in the paper (pretrained models available [here](https://github.com/wgrathwohl/VERA/releases/tag/1.0.0)), run:

```markdown
python train.py --dataset DATASET  # cifar10 or cifar100
                --ent_weight 0.0001  --noise_dim 128  \
                --viz_every 1000 --save_dir /YOUR/SAVE/DIR --data_aug --dropout .3 --thicc_resnet \
                --ckpt_path /PATH/TO/YOUR/MODEL.pt --generator_type vera --n_epochs 200 --print_every 100 \
                --lr .00003 --glr .00006 --post_lr .00003 --batch_size 40 --pg_control .1 \
                --decay_epochs 150 175 --jem  --warmup_iters 2500 --clf_weight 100. --g_feats 256
```

### Evaluation

To evaluate the classifier (on CIFAR10):
```markdown
python eval.py --ckpt_path /PATH/TO/YOUR/MODEL.pt --eval test_clf --dataset cifar_test
```
To do OOD detection (on CIFAR100)
```markdown
python eval.py --ckpt_path /PATH/TO/YOUR/MODEL.pt --eval OOD --ood_dataset cifar_100
```
To generate a histogram of OOD scores.
```markdown
python eval.py --ckpt_path /PATH/TO/YOUR/MODEL.pt --eval logp_hist --datasets cifar10 svhn --save_dir /YOUR/HIST/FOLDER
```
To generate unconditional samples
```markdown
python eval.py --ckpt_path /PATH/TO/YOUR/MODEL.pt --eval uncond_samples --save_dir /YOUR/SAVE/DIR --n_sample_steps 100 --n_steps 40
```
To generate conditional samples
```markdown
python eval.py --ckpt_path /PATH/TO/YOUR/MODEL.pt --eval cond_samples --save_dir /YOUR/SAVE/DIR --n_sample_steps 100 --n_steps 40
```

### Mode counting

Models can be trained by passing in the `--dataset stackmnist` flag. 

Code for counting captured modes of a saved model is available in `mode_counting/stackmnist_mode.py`. 

Code for training the MNIST classifier for counting modes is available in `mode_counting/mnist_classify.py`. 

Hyperparameters may be found in the paper. In particular note that results were reported on MNIST rescaled to 64x64, which can be specified with `--img_size 64`.

## Data

Tabular data for semi-supervised classification must be downloaded manually and placed in `datasets/`.

### HEPMASS

Download `1000_train.csv.gz` and `1000_test.csv.gz` from [here](http://archive.ics.uci.edu/ml/datasets/HEPMASS). Unzip each of these files and place in `datasets/HEPMASS/`.

### HUMAN

Download `UCI HAR Dataset.zip` from [here](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones). Unzip. Rename the resulting folder to `HUMAN/` and place this folder in `datasets/`.

### CROP

Download `data.zip` from [here](https://archive.ics.uci.edu/ml/datasets/Crop+mapping+using+fused+optical-radar+data+set). Unzip. Place the resulting file in `datasets/CROP/`.

### Summary of necessary files
If you want to use all three datasets, the `datasets/` folder should include these files:

```markdown
datasets/
|-- HEPMASS
|   |-- 1000_train.csv
|   |-- 1000_test.csv
|-- HUMAN
|   |-- train
|       |-- X_train.txt
|       |-- y_train.txt
|   |-- test
|       |-- X_test.txt
|       |-- y_test.txt
|-- CROP
|   |-- WinnipegDataset.txt
```

## Acknowledgements
Some code from this repository was adapted from the following repositories:
- [JEM](https://github.com/wgrathwohl/JEM)
- [WideResnet](https://github.com/meliketoy/wide-resnet.pytorch)
- [VAT](https://github.com/lyakaap/VAT-pytorch)

## BibTeX

```markdown
@article{grathwohl2020nomcmc,
  title={No MCMC for me: Amortized sampling for fast and stable training of energy-based models},
  author={Grathowhl, Will and Kelly, Jacob and Hashemi, Milad and Norouzi, Mohammad and Swersky, Kevin and Duvenaud, David},
  journal={arXiv preprint arXiv:2010.04230},
  year={2020}
}
```
