# ASMoE
The source code of our ASMoE.


## Preparation
We train and evaluate our ASMoE using a Tesla A100 GPU with 40 GB memory. <br>
Our code requires the following packages:

> + numpy==1.21.6
> + scikit-learn==1.0.2
> + scipy==1.7.3
> + six==1.17.0
> + torch==1.13.1+cu116
> + tensorboard==2.11.2
> + colorlog==6.10.1
> + pandas==1.3.5


## Usage

We provide one dataset, i.e., Toys `./dataset/Amazon_Toys_and_Games` for reproduction.  <br>
Please download the other three datasets from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). And put the files in `./dataset/` like the following.

```
$ tree
.
├── Amazon_Beauty
│   ├── Amazon_Beauty.inter
│   └── Amazon_Beauty.item
├── Amazon_Toys_and_Games
│   ├── Amazon_Toys_and_Games.inter
│   └── Amazon_Toys_and_Games.item
├── Amazon_Sports_and_Outdoors
│   ├── Amazon_Sports_and_Outdoors.inter
│   └── Amazon_Sports_and_Outdoors.item
└── yelp
    ├── README.md
    ├── yelp.inter
    ├── yelp.item
    └── yelp.user
```

To obtain the first-stage trained ASMoE, run the command`./run_asmoe_s1.sh`. <br>

For the fast reproduction, we provide the trained models of the first stage in `./saved/`. <br>
Run the command`./run_asmoe_s2.sh`. After training and evaluation, check the results in `./run_results/`.


