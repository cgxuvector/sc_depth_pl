from tqdm import tqdm
from imageio import imread, imwrite
from path import Path
import os

from config import get_opts

from SC_Depth import SC_Depth
from SC_DepthV2 import SC_DepthV2

import datasets.custom_transforms as custom_transforms
import matplotlib.pyplot as plt
from skimage.transform import resize

from visualization import *

import IPython.terminal.debugger as Debug


@torch.no_grad()
def main():
    hparams = get_opts()

    env_name = "room_0"
    hparams.input_dir = f"demo/input/habitat_{env_name}/"
    hparams.dataset_name = "nyu"
    hparams.output_dir = "demo/output/"
    hparams.config = "configs/v2/nyu.txt"
    hparams.ckpt_path = "ckpts/nyu_scv2/version_3/epoch=192-val_loss=0.5880.ckpt"

    if hparams.model_version == 'v1':
        system = SC_Depth(hparams)
    elif hparams.model_version == 'v2':
        system = SC_DepthV2(hparams)

    model = system.load_from_checkpoint(hparams.ckpt_path)
    model.cuda()
    model.eval()

    # training size
    if hparams.dataset_name == 'nyu':
        training_size = [256, 320]
    elif hparams.dataset_name == 'kitti':
        training_size = [256, 832]
    elif hparams.dataset_name == 'ddad':
        training_size = [384, 640]

    # normaliazation
    inference_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo(training_size),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize()]
    )

    input_dir = Path(hparams.input_dir)
    output_dir = Path(hparams.output_dir) / \
        'model_{}'.format(hparams.model_version)
    output_dir.makedirs_p()

    if hparams.save_vis:
        (output_dir/'vis').makedirs_p()

    if hparams.save_depth:
        (output_dir/'depth').makedirs_p()

    image_files = sum([(input_dir).files('*.{}'.format(ext))
                      for ext in ['jpg', 'png']], [])
    image_files = sorted(image_files)

    print('{} images for inference'.format(len(image_files)))

    # create figures
    fig, arr = plt.subplots(1, 3, figsize=(12, 8))
    artist_1, artist_2, artist_3 = None, None, None
    for i, img_file in enumerate(tqdm(image_files)):
        img_file = Path(f'demo/input/habitat_{env_name}/0{i+1}.jpg')  # file path
        rgb_img = resize(imread(img_file).astype(np.uint8), (256, 320))  # rgb image
        img = imread(img_file).astype(np.float32)
        tensor_img = inference_transform([img])[0][0].unsqueeze(0).cuda()
        pred_depth = model.inference_depth(tensor_img)

        vis_rgb = rgb_img
        vis_depth = pred_depth.cpu().squeeze(dim=0).squeeze(dim=0).numpy()
        print(f"./demo/input/habitat_{env_name}/0{i+1}.npy")
        vis_gt_depth = resize(np.load(f"./demo/input/habitat_{env_name}/0{i+1}.npy"), [256, 320])

        # rescale the vis_depth
        # vis_depth = vis_depth * (np.median(vis_gt_depth) / np.median(vis_depth))
        print(f"pred depth - {vis_depth.min()} - {vis_depth.max()} ")
        print(f"GT depth - {vis_gt_depth.min()} - {vis_gt_depth.max()}")

        if i == 0:
            arr[0].set_title("RGB")
            artist_1 = arr[0].imshow(vis_rgb)
            arr[0].axis("off")
            vmin = str(np.round(vis_depth.min(), 2))
            vmax = str(np.round(vis_depth.max(), 2))
            arr[1].set_title(f"Pred Depth - {vmin}/{vmax}")
            artist_2 = arr[1].imshow(vis_depth, cmap="gray")
            arr[1].axis("off")
            vmin = str(np.round(vis_gt_depth.min(), 2))
            vmax = str(np.round(vis_gt_depth.max(), 2))
            arr[2].set_title(f"GT Depth - {vmin}/{vmax}")
            artist_3 = arr[2].imshow(vis_gt_depth, cmap="gray")
            arr[2].axis("off")
        else:
            artist_1.set_data(vis_rgb)
            vmin = str(np.round(vis_depth.min(), 2))
            vmax = str(np.round(vis_depth.max(), 2))
            arr[1].set_title(f"Pred Depth - {vmin}/{vmax}")
            artist_2.set_data(vis_depth)
            vmin = str(np.round(vis_gt_depth.min(), 2))
            vmax = str(np.round(vis_gt_depth.max(), 2))
            arr[2].set_title(f"GT Depth - {vmin}/{vmax}")
            artist_3.set_data(vis_gt_depth)
        fig.canvas.draw()
        plt.pause(0.7)


if __name__ == '__main__':
    main()
