#!/usr/bin/env python3

from einops import rearrange
from segment_anything import SamPredictor, sam_model_registry

from conf import settings
# from dataset import *
# from models.discriminatorlayer import discriminator
from dataset import *
from utils import *

# run with command line parameters:
# -dataset cxr -mod sam_adpt -net sam -sam_ckpt Data/Models/sam-med2d_b.pth -encoder vit_b -b 1 -w 0
EXPERIMENT = 'sam-med2d_b'


def main():
    args = cfg.parse_args()
    args.data_path = "./Data/"
    args.dataset = "cxr"
    args.vis = 1
    args.exp_name = EXPERIMENT

    GPUdevice = torch.device('cuda', args.gpu_device)

    assert os.path.exists(args.sam_ckpt)
    net = sam_model_registry["vit_b"](checkpoint=args.sam_ckpt).to(device)
    start_epoch = 0

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    '''segmentation data'''
    nice_train_loader, nice_test_loader = get_dataloader(args)

    tol, (eiou, edice) = validation_sam(args, nice_test_loader, start_epoch, net)
    logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {start_epoch}.')


GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice) * 2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1, 11, (args.b, 7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []


def validation_sam(args, val_loader, epoch, net, clean_dir=True):
    # eval mode
    net.eval()

    predictor = SamPredictor(net)

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0, 0, 0, 0), (0,) * args.multimask_output * 2
    rater_res = [(0, 0, 0, 0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masksw = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            # for k,v in pack['image_meta_dict'].items():
            #     print(k)
            if 'pt' not in pack or args.thd:
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']
            name = pack['image_meta_dict']['filename_or_obj']

            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:, :, buoy: buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[..., buoy:buoy + evl_ch]
                masks = masksw[..., buoy:buoy + evl_ch]
                buoy += evl_ch

                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w ')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w ')
                    imgs = imgs.repeat(1, 3, 1, 1)
                    point_labels = torch.ones(imgs.size(0))

                    imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)

                showp = pt

                mask_type = torch.float32
                ind += 1
                b_size, c, w, h = imgs.size()
                longsize = w if w >= h else h

                if point_labels.clone().flatten()[0] != -1:
                    # point_coords = samtrans.ResizeLongestSide(longsize).apply_coords(pt, (h, w))
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    if (len(point_labels.shape) == 1):  # only one point prompt
                        coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[
                            None, :, :]
                    pt = (coords_torch, labels_torch)

                '''init'''
                if hard:
                    true_mask_ave = (true_mask_ave > 0.5).float()
                    # true_mask_ave = cons_tensor(true_mask_ave)
                imgs = imgs.to(dtype=mask_type, device=GPUdevice)

                '''test'''
                with torch.no_grad():
                    sam_image = imgs.squeeze(axis=0)
                    sam_image = torch.transpose(sam_image, 0, -1)  # make channel last
                    sam_image = torch.transpose(sam_image, 0, 1)  # swap height and width
                    predictor.set_image(sam_image.cpu().numpy())
                    # predictor.set_torch_image(imgs, imgs.shape[:2])
                    pred, probs, _ = predictor.predict(
                        point_coords=pt[0].cpu().numpy().squeeze(axis=0),
                        point_labels=pt[1].cpu().numpy().squeeze(axis=0),
                        return_logits=False)
                    # import itk  # debug
                    # itk.imwrite(itk.image_view_from_array(pred.astype(np.uint8)), "pred.nrrd")
                    pred = torch.Tensor(np.expand_dims(pred, 0)).to(device=GPUdevice)

                    # pred = pred[:, :args.multimask_output, :, :]  # first mask(s)
                    pred = pred[:, 1:2, :, :]  # middle mask
                    # pred = pred[:, -args.multimask_output:, :, :]  # last mask(s)

                    # Resize to the ordered output size
                    pred = F.interpolate(pred, size=(args.out_size, args.out_size))
                    tot += lossfunc(pred, masks)

                    '''vis images'''
                    if ind % args.vis == 0:
                        namecat = 'Test'
                        for na in name[:2

                        ]:
                            img_name = na.split('/')[-1].split('.')[0]
                            namecat = namecat + img_name + '+'
                        vis_image(imgs, pred, masks, os.path.join(args.path_helper['sample_path'],
                                                                  namecat + 'epoch+' + str(epoch) + '.jpg'),
                                  reverse=False, points=showp)

                    temp = eval_seg(pred, masks, threshold)
                    mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot / n_val, tuple([a / n_val for a in mix_res])


if __name__ == '__main__':
    main()
