# ERA: Meta Representation Alignment for Data Bias Mitigation in Recommendations

## Abstract
Recommender systems are widely used to help users discover content of interest. However, due to their reliance on observational user-item interaction data, they often suffer from data bias. Such biases primarily stem from non-random exposure and users’ self-selection behavior, which distort the data distribution and lead to suboptimal performance of recommendation models. Existing debiasing methods, especially those based on loss reweighting strategies, have shown promising empirical results but still lack solid theoretical guarantees. In particular, they struggle to handle the complex, diverse, and often unidentifiable types of bias encountered in real-world scenarios.
In this paper, we revisit the problem of unbiased recommendation from the perspective of data bias and propose a unified debiasing framework that mitigates the effect of bias by aligning the distribution of training data with that of unbiased data collected under randomized exposure. We provide a thorough analysis of the theoretical limitations of existing reweighting methods, and we further propose a principled method, m**E**ta **R**epresentation **A**lignment (ERA), aiming to alleviate the inconsistency between user and item features under different distributions. Extensive experiments on real-world and semi-synthetic datasets demonstrate the effectiveness of ERA.

## Project Introduction
Era is a recommendation system debiasing framework based on meta-learning and adversarial training, aiming to solve the selection bias problem in recommendation systems. By automatically learning sample weights and domain adversarial training, this method can effectively mitigate the impact of biased training data on the recommendation model and improve recommendation performance on unbiased test sets.

## Main Features
- 🎯 Automatic weight learning based on meta-learning, without the need to manually adjust sample weights
- ⚔️ Domain adversarial training to reduce the distribution difference between biased data and uniform data
- 📊 Clustering loss optimization to improve the separability of positive and negative samples
- 🚀 Support for multiple public datasets: Coat, Yahoo!R3, KuaiRand-1K
- 📈 Comprehensive evaluation metrics: MSE, NLL, AUC, Recall, Precision, NDCG

## Environmental Requirements
- Python 3.8+
- PyTorch 1.8+
- CUDA 10.2+
- Dependencies: numpy, pandas, scipy, scikit-learn, tensorboard, cppimport (optional, for accelerating evaluation)

## Install Dependencies
```bash
pip install torch numpy pandas scipy scikit-learn tensorboard cppimport
```

## Dataset Preparation
1. Download the corresponding datasets and place them in the `datasets/` directory
2. Each dataset needs to contain two files:
   - `user.txt`: Biased user feedback data, format: `uid,iid,rating`
   - `random.txt`: Uniformly sampled unbiased data, format: `uid,iid,rating`

### Supported Datasets
| Dataset | Size | Threshold |
|--------|------|------|
| Coat | 290 users, 300 items | 4 |
| Yahoo!R3 | 15400 users, 1000 items | 4 |
| KuaiRand-1K | 1156 users, 3903 items | 1 |

## Startup Commands

### Main Program Entry (Recommended)
The code has been refactored into the `src/` directory. It is recommended to use the new version:
```bash
# Coat dataset
python src/main.py --seed 0 --device 0 --data_name coat --type implicit

# Yahoo!R3 dataset
python src/main.py --seed 0 --device 0 --data_name yahooR3 --type implicit

# KuaiRand-1K dataset
python src/main.py --seed 0 --device 0 --data_name KuaiRand-1K --type implicit
```

### Background Execution Commands
- Coat dataset:
```bash
nohup python -u src/main.py --seed 0 --device 0 --data_name coat --type implicit > logs/coat.log 2>&1 &
```

- Yahoo!R3 dataset:
```bash
nohup python -u src/main.py --seed 0 --device 0 --data_name yahooR3 --type implicit > logs/yahooR3.log 2>&1 &
```

- KuaiRand dataset:
```bash
nohup python -u src/main.py --seed 0 --device 0 --data_name KuaiRand-1K --type implicit > logs/kuairand.log 2>&1 &
```

## Main Parameters Description
| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| --seed | int | 0 | Random seed to ensure reproducible experiments |
| --device | int | 1 | GPU device ID |
| --data_name | str | yahooR3 | Dataset name: yahooR3/coat/KuaiRand-1K |
| --type | str | implicit | Feedback type: implicit/explicit |
| --threshold | int | 4 | Rating binarization threshold, >= threshold is positive sample |
| --debug | bool | False | Whether to enable debug mode |
| --epochs | int | 200 | Maximum training epochs |
| --patience | int | 50 | Early stopping patience, training stops if AUC doesn't rise for this many epochs |

## Project Structure
```
├── src/                    # Source code directory
│   ├── main.py             # Main program entry
│   ├── model.py            # Model definitions (Recommendation model, Discriminator, Weight network)
│   ├── load_dataset.py     # Dataset loading
│   ├── data_loader.py      # Data loaders
│   └── arguments.py        # Parameter parsing
├── utils/                  # Utility functions
│   ├── metrics.py          # Evaluation metrics
│   ├── early_stop.py       # Early stopping mechanism
│   └── ...
├── datasets/               # Datasets directory
├── logs/                   # Logs and saved models directory
├── PlayGround/             # Old experiment code
└── README                  # Project documentation
```

## Expected Results
On the Yahoo!R3 dataset, AUC can reach ~0.78, Recall@5 can reach ~0.81; on the Coat dataset, AUC can reach ~0.70+.

## View in TensorBoard
Losses and metrics during training are automatically written to TensorBoard logs, which can be viewed via the following command:
```bash
tensorboard --logdir logs/
```

## Reference Paper
This method is based on the AutoDebias framework, combining the ideas of meta-learning and adversarial training to solve the selection bias problem in recommendation systems.

