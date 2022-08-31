# Stage II. Joint Optimization with Inverse Rendering With
  
In this stage, we model surface normals, BRDFs, and light visibility of the scene with MLPs. The weights of the MLPs and the lights are jointly optimized to fit the input multiview and multi-light images. We utilize the initial shape extracted from stage I, so please make sure you have trained [Stage I](../stage1) and extracted the initial shapes.

## Training
For training a model from scratch, run 
```bash
## replace `OBJ_NAME`, `GPU_ID`  with your choices.
python train.py --conf confs/OBJ_NAME.conf --gpu GPU_ID
```


## Relighting and Material Editing
Our method jointly estimates surface normals, spatially-varying BRDFs, and lights. After optimization, the reconstructed object can be used for novel-view rendering, relighting, and material editing. You may run the following commands for rendering under environment lighting or material editing.

- For rendering with environment lighting
```bash
## replace `OBJ_NAME`, `GPU_ID` and 'EXPNAME` with your choices. 
## You may specify the envmap ID (there are some demo envmaps provided under "envmap" folder, please make sure you have downloaded and extracted the envmaps).
python eval.py --gpu GPU_ID --obj_name OBJ_NAME --expname EXPNAME --render_envmap 
## arguments
# --envmap_id ENVMAP_ID       # specify the envmap ID (default: 3)
# --envmap_path ENVMAP_PATH   # specify the path of envmaps (default: ./envmap)
# --envmap_scale SCALE        # modify the light intensity scale (default: 1)
```

- For material editing
```bash
## replace `OBJ_NAME`, `GPU_ID` and 'EXPNAME` with your choices. 
## For albedo editing, enable `--edit_albedo`; for specular editing, enable `--edit_specular`.
python eval.py --gpu GPU_ID --obj_name OBJ_NAME --expname EXPNAME 
## arguments
# --edit_albedo       # enable albedo editing
# --color COLOR       # specify the new color using <Hex Color Code>, e.g., "#E24A0F". if not specified, a random color will be used.
# --edit_specular     # enable specular editing
# --basis BASIS       # specify the new basis ID, e.g., "3". if not specified, a random basis will be used.
```

```bash
## other optional arguments
# --exps_folder EXP_FOLDER        # specify exp_folder (default: ./out)
# --test_out_dir TEST_OUT_DIR     # test_out_dir (default: ./test_out)
# --save_npy                      # save npy files
# --timestamp TIMESTAMP           # specify the timestamp (default: latest)
# --checkpoint CHECKPOINT         # specify the checkpoint (default: latest)
# --light_batch N_LIGHT           # modify light batch according to your GPU memory (default: 64)
```
