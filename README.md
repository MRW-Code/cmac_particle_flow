# cmac_particle_flow

Populate the "images" and "external_test_set" as needed. \
Make empty dir for "aug_images" folder and the "split_test_images". \
Code will empty files as needed for you so long as they are present. \
File names as shown and appear in main "cmac_particle_flow" dir. \
May need to add the empty directories manually. 

Train a model: \
Run ttv_main.py and select the split index and split_factor. \
split_index = The index of the smaller crop to take for the validation set. \
split_factor = The divisible factor applied to the width and height. Changed the number/size of the quadrants to train on. \

Inference: \
Run do_inference.py and pass the learner you wish to use. \
Make sure learner is the .pkl NOT the .pth version. \
Now takes the split_factor variable. Make sure this is the same as the model used so the same preprocessing is applied to test set. 
