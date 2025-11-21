# LG-Transformer
[![python](https://badge.ttsalpha.com/api?icon=python&label=python&status=3.9.20)](https://example.com)  [![pytorch](https://badge.ttsalpha.com/api?icon=pytorch&label=pytorch&status=2.4.1&color=yellow)](https://example.com)  [![cuda](https://badge.ttsalpha.com/api?icon=nvidia&label=cuda&status=11.8&color=green)](https://example.com)


## Project Overview
Green fuels are essential for decarbonizing transportation sectors,  requiring accurate prediction of multiple physicochemical properties to optimize engine performance and emissions. Although artificial intelligence-based models demonstrate significant potential to accelerate fuel design, most existing methods cannot utilize the internal and external information within and between fuel molecules with interpretability, limiting their generalizability and multi-objective capabilities. To address these challenges, a novel deep learning framework, the learned graph feature fusion transformer (LG-Transformer), is proposed. This unified framework integrates contrastive learning-based graph construction with graph convolutional networks (GCNs) and Transformer encoders to simultaneously predict multiple key physicochemical properties critical for green fuel design. Supporting this effort, a comprehensive fuel property database is developed, containing 1850 diverse molecules across 26 chemical classes, each annotated with 17 key physicochemical properties relevant to engine performance. LG-Transformer achieves state-of-the-art performance with an average $R^2$ of 0.933, significantly outperforming conventional quantitative structure–property relationship models and deep learning baselines. Additionally, interpretability analyses via integrated gradients reveal underlying molecular structure–property relationships. Overall, this work establishes a powerful, generalizable, and interpretable framework that significantly accelerates virtual screening and rational design of advanced low-carbon fuels, bridging fundamental understanding and practical application to advance the discovery and optimization of next-generation sustainable fuels.

![framework](./image.png)

## Installation


### 1. Cloning the Project

First, you need to clone the project repository from GitHub to your local machine. You can do this by running the following command in your terminal:

```bash
git clone https://github.com/furyIndex/AI4fuel.git
```

This command will create a copy of the   project in your current working directory.

### 2. Setting Up the Environment

After cloning the project, the next step is to set up the project environment. This project uses Conda, a popular package and environment management system. To create the environment with all the required dependencies, navigate to the project directory and run:

```bash
cd AI4fuel
conda env create -f environment.yml
```

This command will read the environment.yml file and create a new Conda environment with the name specified in the file. It will also install all the dependencies listed in the file.
For installing the algos, you should use
```bash
python setup.py build_ext --inplace
``` 


### 3. Activating the Environment

Once the environment is created, you need to activate it. To do so, use the following command:

```bash
conda activate ai4fuel
```

Replace **ai4fuel** with the actual name of the environment, as specified in the **environment.yml** file.






## Training

Before training, you need to copy the dataset corresponding to this project to the ``./data`` directory. [Here](https://drive.google.com/file/d/11w5gG3wpER595AhroFRgGlBFD78njCzc/view?usp=drive_link) contains all databases used in this project.

Usage:
```bash
./train.sh
```

## Reproduce

Usage:
```bash
./reproduce.sh
```


## Predict

Usage:
```bash
./predict.sh
```




## Demo

### train
```bash
./train.sh
```

The expected output is as follows:

```
Fold (dir=fold_1): Test R2 = 0.9788  (seq_len=50, input_dim=606)
------------------------------------------------------------
Sheet: EOV | R2s: [0.9788]
Mean R2: 0.9788
Std  R2: 0.0

================================================================================
Summary of the completed training worksheets：
 - HOV  Mean R2 = 0.9788  Std = 0.0000  folds = [0.9788]
================================================================================
```

### reproduce
```bash
./reproduce.sh
```

The expected output is as follows:

```
[test] Reproduce R2 = 0.9788 (seq_len=50, input_dim=606)
Saved: ./train_kfold/lg_transformer_kfold_runs/sheet_EOV/fold_1/predictions.csv
```

### predict
```bash
./predict.sh
```

The expected output is as follows:

```
Using outer_train (inner_train + inner_val) as graph anchor, total 442 samples.
Loaded 442 samples as graph anchor from outer_train (inner_train + inner_val).
Loaded 62 new samples for prediction.
[Prediction Done] seq_len=50, input_dim=606, samples=62
[Evaluation] R² on new data = 0.905895
Saved predictions to: ./train_kfold/lg_transformer_kfold_runs/sheet_EOV/fold_5/./predictions_with_HOV_fold1.csv
```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

