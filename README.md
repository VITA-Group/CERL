# CERL: A Unified Optimization Framework forLight Enhancement with Realistic Noise
Code used for [CERL: A Unified Optimization Framework for Light Enhancement with Realistic Noise](https://arxiv.org/abs/2108.00478). 

## Representitive Results
![representive_results](/imgs/teaser.png)

## Overal Architecture
![architecture](/imgs/cerl.png)

## Training
* Download LOL dataset provided by this [github respository](https://github.com/flyywh/CVPR-2020-Semi-Low-Light). Put them in ./data
* Download pre-trained DnCNN parameters from this [github respository](https://github.com/cszn/KAIR). Put them in ./original_model
```
python train.py
```
The parameters of fine-tuned denoising network (DnCNN) will be saved in ./checkpoints for testing.

## Testing
* Download pre-trained model parameters from [Google Drive](https://drive.google.com/drive/folders/1sHTx1ksZlQ2HSHmHt8UgcZK81NRekJjL?usp=sharing). Put them in ./checkpoints
* Use pre-trained EnlightenGAN model to generate enhanced noisy images. Put them in ./test_imgs
```
python test.py
```

## Realistic Low-light Mobile Photography (RLMP)
RLMP is a new benchmark of the real-world image low-light enhancement task. Images are captured by different types of smartphones. Compared with previous low-light datasets, images in RLMP typically display much more noticeable ISO noise, which complements the existing benchmarks and significantly challenges current enhancement methods. You can get RLMP from this [link](https://drive.google.com/drive/folders/1b36CORsUF8bA04Mrenvvv6cTEiaaji6Y?usp=sharing).

## Citation
if you find this repo is helpful, please cite
```
@article{chen2021cerl,
  title={CERL: A Unified Optimization Framework for Light Enhancement with Realistic Noise},
  author={Zeyuan Chen and Yifan Jiang and Dong Liu and Zhangyang Wang},
  journal={arXiv preprint arXiv:2108.00478},
  year={2021}
}
```
