#!/usr/bin/env python3

import function
# from dataset import *
# from models.discriminatorlayer import discriminator
from dataset import *
from utils import *


def main():
    args = cfg.parse_args()
    lung_ai_path = 'M:/Dev/CXR/LungAI/'
    args.weights = lung_ai_path + "Data/Models/cxr_v5.pth"
    args.sam_ckpt = lung_ai_path + "Data/Models/cxr_v5.pth"
    args.data_path = "./Data/"
    args.dataset = "cxr"
    args.vis = 1
    args.exp_name = "cxr_v5"

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

    '''load pretrained model'''
    assert args.weights != 0
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    start_epoch = checkpoint['epoch']

    net.load_state_dict(checkpoint['state_dict'])


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
