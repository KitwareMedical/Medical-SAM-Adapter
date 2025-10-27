#!/usr/bin/env python3

import function
# from dataset import *
# from models.discriminatorlayer import discriminator
from dataset import *
from utils import *

# run with command line parameters:
# -dataset cxr -mod sam_adpt -net sam -sam_ckpt Data/Models/sam_vit_b_01ec64.pth -encoder vit_b
EXPERIMENT = 'vanilla_sam_vit_b_01ec64'


def main():
    args = cfg.parse_args()
    args.data_path = "./Data/"
    args.dataset = "cxr"
    args.vis = 1
    args.exp_name = EXPERIMENT

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)


    assert os.path.exists(args.sam_ckpt)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(args.sam_ckpt, map_location=loc)
    start_epoch = 0

    net.load_state_dict(checkpoint, strict=False)


    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    '''segmentation data'''
    nice_train_loader, nice_test_loader = get_dataloader(args)

    '''begain valuation'''
    best_acc = 0.0
    best_tol = 1e4

    net.eval()

    tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, start_epoch, net)
    logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {start_epoch}.')


if __name__ == '__main__':
    main()
