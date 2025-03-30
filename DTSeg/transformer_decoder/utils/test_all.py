#! /usr/bin/python
# -*- coding: utf-8 -*-
# Create by zcuncun @ 2021/03/18 11:08:26
'''
gpu_train.py
'''

import argparse, logging
import torch
import time
import random


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--device',
        type=str,
        required=True,
        help='device',
    )

    parser.add_argument(
        '--time',
        type=int,
        default=30000,
        help='运行时间',
    )

    parser.add_argument(
        '--gpu_mem',
        default=12,
        type=int,
        help='显存占用大小',
    )

    parser.add_argument(
        '--cpu_mem',
        default=1,
        type=int,
        help='显存占用大小',
    )


    parser.add_argument(
        '--usage',
        default=50,
        type=int,
        help='使用率',
    )

    args = parser.parse_args()

    train_device = 'cuda:{}'.format(args.device)
    gpu_run = torch.rand([64, 3, 512, 512], device=train_device)
    gpu_mem = torch.rand([(args.gpu_mem - 1) * (1 << 28)], device=train_device)
    cpu_run = torch.rand([8, 3, 256, 256], device='cpu')
    cpu_mem = torch.rand([args.cpu_mem * (1 << 28)], device='cpu')
    start_time = time.time()
    t = random.randint(10, 20)
    while True:
        t0 = time.time()
        a = 0
        while time.time() - t0 < (t - 2):
            time.sleep(0.01)
            for i in range(args.usage):
                torch.sin(gpu_run)
        while time.time() - t0 < t:
            a = 1
        if args.time > 0 and time.time() - start_time > args.time:
            break


if __name__ == '__main__':
    main()