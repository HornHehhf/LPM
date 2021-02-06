# Layer-Peeled Model
This is the code repository for the ArXiv paper [Layer-Peeled Model: Toward Understanding Well-Trained Deep Neural Networks](https://arxiv.org/pdf/2101.12699.pdf).
If you use this code for your work, please cite
```
@article{fang2021layer,
  title={Layer-Peeled Model: Toward Understanding Well-Trained Deep Neural Networks},
  author={Fang, Cong and He, Hangfeng and Long, Qi and Su, Weijie J},
  journal={arXiv preprint arXiv:2101.12699},
  year={2021}
}
```
## Installing Dependencies
Use virtual environment tools (e.g miniconda) to install packages and run experiments\
python==3.6.7\
pip install -r requirements.txt

## Code Organization

The code is organized as follows:
- optimization.py (solving the relaxed convex optimization program of the nonconvex Layer-Peeled Model)
- train_models.py (training deep learning models for minority collapse)
- minority_collapse.py (analyzing minority collapse in deep neural networks)
- train_models_oversampling.py (training deep learning models for the oversampling algorithm)
- minority_collapse_oversampling.py (analyzing minority collapse in the oversampling algorithm)


## Change the Dir Path

Change the /path/to/working/dir to your working dir.


## Reproducing deep learning experiments
To reproduce the experiments for [neural collapse](https://www.pnas.org/content/117/40/24652.short)
```
python train_models.py data_option=$data_option model_option=$model_option t1=10 R=1
python minority_collapse.py data_option=$data_option model_option=$model_option t1=10 R=1
python train_models.py data_option=cifar10 model_option=VGG13 t1=10 R=1 (an example)
python minority_collapse.py data_option=cifar10 model_option=VGG13 t1=10 R=1 (an example)
```


To reproduce the experiments for minority collapse (Figure 4)
```
python train_models.py data_option=$data_option model_option=$model_option t1=$t1 R=$R
python minority_collapse.py data_option=$data_option model_option=$model_option t1=$t1 R=$R
python train_models.py data_option=cifar10 model_option=VGG13 t1=5 R=10 (an example)
python minority_collapse.py data_option=cifar10 model_option=VGG13 t1=5 R=10 (an example)
```

To reproduce the experiments for the oversampling algorithm (Figure 6)
```
python train_models_oversampling.py data_option=$data_option model_option=$model_option t1=$t1 R=1000 weight_ratio=$weight_ratio
python minority_collapse_oversampling.py data_option=$data_option model_option=$model_option t1=$t1 R=1000 weight_ratio=$weight_ratio
python train_models_oversampling.py data_option=cifar10 model_option=VGG13 t1=5 R=1000 weight_ratio=10 (an example)
python minority_collapse_oversampling.py data_option=cifar10 model_option=VGG13 t1=5 R=1000 weight_ratio=10 (an example)
```

## Reproducing experiments for Layer-Peeled Model
In this part, you need to install cvxpy==1.1.7 successfully (in a new environment), which might be incompatible with dependencies in deep learning experiments. \
To reproduce the experiments for the relaxed convex optimization program of the Layer-Peeled Model (Figure 3)
```
python optimization.py
```