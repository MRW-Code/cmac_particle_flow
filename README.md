# cmac_particle_flow

Populate the images folder and the external test set, make sure the rest are present but empty. \n
Code will empty for you so long as they are present. 

Train a model:
Run ttv_main.py and select the split index and split_factor. 
split_index = The index of the smaller crop to take for the validation set.
split_factor = The divisible factor applied to the width and height. Changed the number/size of the quadrants to train on. 

Inference:
Run do_inference.py and pass the learner you wish to use. 
Make sure learner is the .pkl NOT the .pth version. 
Now takes the split_factor variable. Make sure this is the same as the model used so the same preprocessing is applied to test set. 
