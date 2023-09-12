# cmac_particle_flow
This repository contains the code and data supporting the publication titled "Predicting pharamceutical powder flow from microscopy images using deep learning" published in RSC Digital Discovery (<https://doi.org/10.1039/D2DD00123C>). All work was carried out on a Linux (Ubuntu) machine and as such no guarentee can be offered for system compatibility with other operating systems. Please note that this code is not under active development and as such comes with no guarentee, however should I be made aware of any issues, I will endevour to resolve them as fast as possible. If this work has proved useful, then I ask that you consider citing the orgininal paper.

# Data
The images used for this work are provided in the `images` directory of this repository, with their respective labels being captured by the sub-directories within the `images` parent. 

# Env Set Up
```bash
# Install venv and dependencies
sudo apt-get install python3-venv
# Create working directory
mkdir {chosen working directory}
cd chosen_dir
# Make virtual environment
python3 -m venv env_name
# Activate virtual environment
source env_name/bin/activate
# install packages
pip install -r requirements.txt
```

# Running the code
To use the program, run:
```
python3 ttv_main.py [arguments]
```

## List of Command Line Arguments:

1. **--gpu_idx** : Specifies the GPU index.
    - Choices: 0, 1, 2, 3, 4, 5
    - Default: 0
    ```
    --gpu_idx 2
    ```

2. **-s/--split_factor** : The factor by which the data will be split.
    - Choices: 0 to 59 (inclusive)
    - Default: 3
    ```
    -s 5
    ```

3. **-b/--batch_size** : The size of the batch for processing.
    - Choices: 0 to 999 (inclusive)
    - Default: 8
    ```
    -b 32
    ```

4. **--no_augs** : If present, disables augmentations.
    - Default: False
    ```
    --no_augs
    ```

5. **--make_figs** : If present, generates figures.
    - Default: False
    ```
    --make_figs
    ```

6. **--from_scratch** : If present, starts training from scratch.
    ```
    --from_scratch
    ```

7. **-v/--cv_method** : The cross-validation method to use.
    - Choices: kfold, crop_fold, split_first, ttv, kfold_ttv
    - Default: split_first
    ```
    -v kfold
    ```

8. **-g/--grad_accum** : Specifies the gradient accumulation.
    - Choices: 0 to 49 (inclusive)
    - Default: 1
    ```
    -g 4
    ```

9. **-m/--model** : Specifies the model type.
    - Choices: 
        - convnext_tiny_in22k
        - resnet18
        - convnext_tiny
        - convnext_small
        - resnet50
        - vit_tiny_patch16_384
        - swinv2_tiny_window16_256
        - swinv2_cr_tiny_384
        - vit_small_patch16_384
        - vit_small_patch32_384
        - swinv2_cr_tiny_384
        - swinv2_base_window12to24_192to384_22kft1k
    - Default: resnet18
    ```
    -m resnet50
    ```

10. **--no_resize** : If present, disables image resizing.
    ```
    --no_resize
    ```
    
