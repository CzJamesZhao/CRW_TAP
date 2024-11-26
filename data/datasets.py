import logging
import random
import time

import torch
import torch.distributed as dist
from torch.utils.data.dataloader import default_collate

from data.kubric import KubricDataloader
from data.kinetics import KineticsDataset

logger = logging.getLogger(__name__)

def collate_fn(batch):
    return default_collate(batch)


def make_data_sampler(local_rank, dataset, shuffle=True):
    torch.manual_seed(0)
    if local_rank in [-2, -1]:
        if shuffle:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    else:
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)

    return sampler


def get_data_loader(args, training, val_during_train=False):
    st = time.time()

    IMG_SIZE1 = (args.img_size1_h, args.img_size1_w)
    IMG_SIZE2 = (args.img_size2_h, args.img_size2_w)

    traindir = args.data_path

    if args.dataset_type == "kinetics":
        traindir = "/home/zhaochenzhi/CRW_TAP/datasets/kinetics700-2020"
        # 根据数据集类型加载数据
        dataset = KineticsDataset(
            use_frame_transform=True,
            img_size1=IMG_SIZE1,
            img_size2=IMG_SIZE2,
            split="train",
            root=traindir,
            no_of_frames=args.no_of_frames,
            random_seed=args.seed,
            aug_setting=args.data_aug_setting, #默认"setting2"
            training=training,
        )

        logger.info(f"Dataset loading took {str(time.time() - st)}")

        logger.info("Creating data loaders")
        sampler = make_data_sampler(args.local_rank, dataset, shuffle=training)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.eval_batch_size
            if args.eval_only
            else args.train_batch_size,
            sampler=sampler,
            # num_workers=args.workers, # 这里是4，但可以升到8/16看看，应该更快
            num_workers = 8,
            pin_memory=True, # 将数据加载到内存锁定区域，以加快数据传输到 GPU 的速度
            collate_fn=collate_fn,  # 自定义的批次数据整理函数，用于将多个样本合并成一个批次
            worker_init_fn=random.seed(args.seed), # 设置每个进程的随机种子，确保数据增强的随机性一致
        )

    elif args.dataset_type == "kubric":
        split = "train" if training else "validation"
        batch_size = args.eval_batch_size if args.eval_only else args.train_batch_size

        if args.local_rank not in [-2, -1]:
            args.world_size = dist.get_world_size() # 所有进程的总数，用于分布式训练，这里应该是4，可以打印看一看
        else:
            args.world_size = -1
        # import ipdb
        # ipdb.set_trace()

        if args.random_frame_skip: # 默认False
            # 帧长度差异参数，用于处理视频帧的跳帧操作
            frame_len_diff = -1
        else:
            frame_len_diff = 1
        
        data_loader = KubricDataloader(
            use_frame_transform=True,
            img_size1=IMG_SIZE1,
            img_size2=IMG_SIZE2,
            split=split,
            root=traindir,
            no_of_frames=args.no_of_frames,
            frame_len_diff=frame_len_diff, # 用于控制在选择帧时的跳跃幅度,默认为1
            random_frame_skip=args.random_frame_skip,
            aug_setting=args.data_aug_setting,
            random_seed=args.seed,
            batch_size=batch_size,
            shuffle=False,
            worker_id=args.local_rank,
            # num_workers=args.world_size, # 同上，这里好像是4，可以升到8-16看看
            num_workers = 8,
            num_parallel_point_extraction_calls=args.workers, # 特定于“点提取”任务的并行度，这里设置为分布式训练时的进程数量，这里为4
            training=training,
        )

        logger.info(f"Dataset loading took {str(time.time() - st)} secs")
    else:
        raise NotImplementedError

    return data_loader, args
