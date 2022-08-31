# Initial Lighting and Normal Estimation

In this preprocessing stage, we use the pretrained [SDPS-Net](https://github.com/guanyingc/SDPS-Net) to obtain the coarse lighting/normal estimations from the uncalibrated multi-view multi-light input images. 

The coarse normal estimation will be used to regularize the normals derived from the neural radiance field to improve surface details. The coarse light direction estimation will be used as intialization of the Stage II for joint optimization. The coarse light intensity estimation is optional and could be used if the input images are lit by light sources of different intensity. The output of predicted normal/lighting will be defaultly saved under each dataset folder (named `sdps_out` or `sdps_out_l{N}`).


## Get the Initial Lighting/Normal Estimation
```bash
## replace `GPU_ID` and `OBJ_NAME` with your choices.
CUDA_VISIBLE_DEVICES=GPU_ID \
python test.py \
    --retrain data/models/LCNet_CVPR2019.pth.tar \
    --retrain_s2 data/models/NENet_CVPR2019.pth.tar \
    --benchmark UPS_Custom_Dataset \
    --bm_dir ../dataset/OBJ_NAME

## other optional arguments
# --train_light TRAIN_LIGHT     # specify the train_light
# --light_intnorm_gt            # use images normalized by GT light intensity
```

## Citation
This submodule is adapted from **[SDPS-Net: Self-calibrating Deep Photometric Stereo Networks, CVPR 2019 (Oral)](http://guanyingc.github.io/SDPS-Net/)**, which addresses the problem of learning-based _uncalibrated_ photometric stereo for non-Lambertian surface. If you find this code or the provided models useful in your research, please consider cite: 
```
@inproceedings{chen2019SDPS_Net,
  title={SDPS-Net: Self-calibrating Deep Photometric Stereo Networks},
  author={Chen, Guanying and Han, Kai and Shi, Boxin and Matsushita, Yasuyuki and Wong, Kwan-Yee K.},
  booktitle={CVPR},
  year={2019}
}
```
