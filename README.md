# GDPL
Codes for the paper "Guided Dialog Policy Learning: Reward Estimation for Multi-Domain Task-Oriented Dialog", and you can find our paper at [arxiv](https://arxiv.org/abs/1908.10719)

Cite this paper :

```
@inproceedings{takanobu2019guided,
  title={Guided Dialog Policy Learning: Reward Estimation for Multi-Domain Task-Oriented Dialog},
  author={Takanobu, Ryuichi and Zhu, Hanlin and Huang, Minlie},
  booktitle={EMNLP-IJCNLP}
  year={2019}
}
```

## Data

unzip [zip](https://drive.google.com/open?id=18TYwvoA1viGtOPbj-15KlGJYxXOjnhrr) under `data` directory, or simply running

```
sh fetch_data.sh
```

the pre-processed data are under `data/processed_data` directory

- data preprocessing will be automatically done if `processed_data` directory does not exists when running `main.py`

### Use

the best trained model that interacts with `agenda` is under `data/agenda` directory

```
python main.py --test True --load data/agenda/best > result.txt
```

## Run

Command

```
OMP_NUM_THREADS=1 python main.py {--[option1]=[value1] --[option2]=[value2] ... }
```

Change the corresponding options to set hyper-parameters:

```python
parser.add_argument('--log_dir', type=str, default='log', help='Logging directory')
parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
parser.add_argument('--save_dir', type=str, default='model', help='Directory to store model')
parser.add_argument('--load', type=str, default='', help='File name to load trained model')
parser.add_argument('--load_user', type=str, default='', help='File name to load user simulator')
parser.add_argument('--pretrain', type=bool, default=False, help='Set to pretrain')
parser.add_argument('--test', type=bool, default=False, help='Set to inference')
parser.add_argument('--config', type=str, default='multiwoz', help='Dataset to use')
parser.add_argument('--simulator', type=str, default='agenda', help='User simulator to use')

parser.add_argument('--epoch', type=int, default=32, help='Max number of epoch')
parser.add_argument('--save_per_epoch', type=int, default=5, help="Save model every XXX epoches")
parser.add_argument('--process', type=int, default=16, help='Process number')
parser.add_argument('--batchsz', type=int, default=32, help='Batch size')
parser.add_argument('--batchsz_traj', type=int, default=1024, help='Batch size to collect trajectories')
parser.add_argument('--print_per_batch', type=int, default=400, help="Print log every XXX batches")
parser.add_argument('--update_round', type=int, default=5, help='Epoch num for inner loop of PPO')
parser.add_argument('--lr_rl', type=float, default=3e-4, help='Learning rate of dialog policy')
parser.add_argument('--lr_irl', type=float, default=1e-3, help='Learning rate of reward estimator')
parser.add_argument('--lr_simu', type=float, default=1e-3, help='Learning rate of user simulator')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted factor')
parser.add_argument('--epsilon', type=float, default=0.2, help='Clip epsilon of ratio r(theta)')
parser.add_argument('--tau', type=float, default=0.95, help='Generalized advantage estimation')
parser.add_argument('--anneal', type=int, default=5000, help='Max steps for annealing')
parser.add_argument('--clip', type=float, default=0.02, help='Clipping parameter on WGAN')
```

We have implemented *distributed PPO* for parallel trajectory sampling. You can set ```--process``` to change the number of multi-process, and set ```--batchsz_traj``` to change the number of trajectories each process collects before one update iteration.

The default user simulator is *agenda*, you can set ```--simulator neural``` to use *VHUS*

### pretrain

```
python main.py --pretrain True --save_dir model_agenda
```

**NOTE**: please pretrain the model first

### train

```
python main.py --load model_agenda/best --lr_rl 1e-4 --lr_irl 1e-4 --epoch 16
```
**NOTE**: set ```--load_user``` when using *VHUS* as the simulator

### test

```
python main.py --test True --load model_agenda/best > result.txt
```

## Requirements

python 3

pytorch 1.0

