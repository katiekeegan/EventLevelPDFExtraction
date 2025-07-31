from torch import distributed as dist
import os
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets import *
from utils import triplet_theta_contrastive_loss

def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size):
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
