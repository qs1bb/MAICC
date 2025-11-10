### Install Python environment

Install Python environment with conda:

```bash
conda create -n MAICC python=3.8 -y
conda activate MAICC 
pip install pip==24.0
pip install setuptools==63.2.0
pip install wheel==0.38.4
pip install -r requirements.txt
pip install -e lb-foraging/
pip install -e smacv2/
```

### Install StarCraft

```bash
bash install_sc2.sh
```

### Run experiments

Due to the limitation of attachment size of 50MB, we only provide the dataset of lbf_small_train.sh (LBF: 7x7-15s) here. We will provide datasets for all tasks in the future.

```bash
bash lbf_small_train.sh
bash lbf_train.sh
bash sc2v2_train.sh
```
