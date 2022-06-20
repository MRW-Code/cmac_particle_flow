from src.utils import args
import numpy as np
from src.image_splitting import ImageSplitter
from src.image_augmenting import ImageAugmentor
from src.helpers import paths_from_dir, make_needed_dirs, delete_file
from src.training import kfold_model, split_first_model

'''
   This script caried out the testing to see how far we could split the images
   before the properties of the bulk material were no longer represented by the 
   sample image. 

   Here, in the splitter class:
   Split_factor controls the number of images to split into. 
   0 uses the native images. 1 splits them in half vertically. For n where n is greater than 1,
   there will be n**2 images e.g split factor = 3 gives 9 images. 

   val_idx is the index of the split to be used as testing. It can take any value up to the
   number of crops of the original image (uses zero indexing). 
   '''

if __name__ == '__main__':
    try:
        if args.from_scratch:
            delete_file('./aug_images')
            delete_file('./split_images')
            make_needed_dirs()
    except:
        make_needed_dirs()
    img_paths = paths_from_dir('./images')

    if args.cv_method == 'split_first':
        split_first_model(n_splits=5, img_paths=img_paths)

    elif args.cv_method == 'kfold':
        if args.from_scratch:
            splitter = ImageSplitter(img_paths=img_paths, split_factor=args.split_factor, val_idx=None)
            splitter.save_split_images()
        kfold_model(n_splits=5)

##### ABOVE HERE WORKS AND TESTED, BELOW HERE NOTHING TESTED YET

    elif args.cv_method == 'crop_fold':
        raise NotImplementedError('Not added this yet')
        # splitter = ImageSplitter(img_paths=img_paths, split_factor=args.split_factor, val_idx=0)
        # raw_train, raw_val = splitter.do_splitting()


    else:
        raise NotImplementedError('No Such CV method')




    # # check here if we need a val split here for when val_idx is none
    # if raw_val is None:
    #     # then here we do kfold
    #
    # else:
    #     # then here we do our kfold with val_idx
    #
    # ## move this to the train functions
    # if not args.no_augs:
    #     aug = ImageAugmentor(save_path='./aug_images', training_data=raw_train)
    #     test = aug.do_augs()







    # make split factor and val_idx args
