
## Latent Distribution Adjusting

This is an unofficial implementations for Latent Distribution Adjusting for Face Anti-Spoofing. 
To refer the paper please click [here](https://arxiv.org/abs/2305.09285).

## How to use

### Prepare your dataset

Put your dataset in the folder ``data/``, and corresponding [dataset.py] in ``datasets/``(for instance, define 
*MyDataset* in ``datasets/dataset``), and import your customized dataset in [trainer.py].

### Train

Use *Bash train.sh* to train LDA model. The model consists many hyperparameters, which is stored in ``hyp/LDA.json``. 
During training model's parameter will be automatically stored in ``results/`` on every epoch, denoted with validation loss.
