# Stage I. Initial Shape Modeling

In the first stage, we optimize a neural radiance field with surface normal regularizations to represent the object shape. We utilize the normal predicted by [SDPS-Net](https://github.com/guanyingc/SDPS-Net) as normal supervision, so please make sure you have obtain the coarse normal estimation in [Preprocessing](../preprocessing) before training this part.  
Meanwhile, since we capture images illuminated by different light directions, we use the light-averaged images for stage I training. Please make sure you have calculate the light-averaged images using `light_avg.py`.
  
## Training
For training a model from scratch, run
```bash
##  replace `GPU_ID`, `OBJ_NAME`  with your choices.
python train.py configs/OBJ_NAME.yaml --gpu GPU_ID
```

For convenience, we pre-extract the surface/normal/visibility before training stage II. After training this stage, please run the following command to extract the shapes for each view.

```bash
##  replace `GPU_ID`, `OBJ_NAME`, `EXPNAME` with your choices.
##  add --vis_plus for better visibility optimization (need more disk space)
python shape_extract.py --gpu GPU_ID --obj_name OBJ_NAME --expname EXPNAME --visibility --vis_plus 
## other optional arguments
# --exp_folder EXP_FOLDER         # specify exp_folder (default: ./out)
# --test_out_dir TEST_OUT_DIR     # specify test_out_dir (default: ./exps_shape)
# --load_iter N_ITER              # load from specific iteration
# --visualize                     # visualize the extracted normal/surface/visibility
# --chunk N_CHUNK                 # modify according to your GPU memory (default: 32000)
```

## Test
If you are interested in testing on the first stage, we also provide testing and mesh-extraction codes (adapted from [UNISURF](https://github.com/autonomousvision/unisurf)).
- Output results (e.g.,normal) of test set.
    ```bash
    ##  replace `GPU_ID`, `OBJ_NAME`, `EXPNAME` with your choices.
    python eval.py --gpu GPU_ID --obj_name OBJ_NAME --expname EXP_NAME
    ## other optional arguments
    # --exp_folder EXP_FOLDER         # specify exp_folder (default: ./out)
    # --test_out_dir TEST_OUT_DIR     # specify test_out_dir (default: ./test_out)
    # --load_iter N_ITER              # load from specific iteration
    # --save_npy                      # save npy files
    ```

- Extract mesh from a trained model
  
    Before extracting the mesh, please first compile the extension modules by
    ```bash
    python setup.py build_ext --inplace
    ```
    For quickly extracting a mesh from a trained model, run
    ```bash
    ##  replace `GPU_ID`, `OBJ_NAME`, `EXPNAME` with your choices.
    python extract_mesh.py --gpu GPU_ID --obj_name OBJ_NAME --expname EXP_NAME
    ## other optional arguments
    # --exp_folder EXP_FOLDER         # specify exp_folder (default: ./out)
    # --test_out_dir TEST_OUT_DIR     # specify test_out_dir (default: ./test_out)
    # --load_iter N_ITER              # load from specific iteration
    # --mesh_extension EXTENSION      # choose mesh extension (obj or ply, default: obj)
    ```
