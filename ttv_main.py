import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import glob
torch.cuda.empty_cache()

from models.fastai_prep import FastAIPrep
from fastai.vision.all import *
from fastai.distributed import *

def empty_file(path):
    files = glob.glob(path + '/*')
    for file in files:
        os.remove(file)

empty_file('./aug_images')
empty_file('./split_test_images')

prep = FastAIPrep('./images', 0, 2, './aug_images', './split_test_images')
prep.check_test_train_data()
df = prep.get_fastai_df()
print('done')

tfms = None
dls = ImageDataLoaders.from_df(df,
                               seed=None,
                               fn_col=0,
                               folder=None,
                               suff='',
                               label_col=1,
                               label_delim=None,
                               valid_col=2,
                               item_tfms=None,
                               batch_tfms=tfms,
                               bs=16,
                               val_bs=None,
                               shuffle=True,
                               device=None)

learn = cnn_learner(dls, resnet18, metrics=[accuracy]).to_fp16()

learn.fine_tune(200, 0.00001, cbs=[SaveModelCallback(fname='./best_cbs_200'),
                                  ReduceLROnPlateau(monitor='valid_loss',
                                                    min_delta=0.1,
                                                    patience=2)])


# with learn.distrib_ctx(sync_bn=False): learn.fine_tune(100, 0.0001, cbs=[SaveModelCallback(fname='./best_cbs'),
#                                  ReduceLROnPlateau(monitor='valid_loss',
#                                                    min_delta=0.1, patience=2)])


learn.export('./code_saves/trained_model_200.pkl')

learn.recorder.plot_loss()
plt.savefig('./code_saves/training_plot_200.png')

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
plt.savefig('./conf_mtrx_200')


# need to load the test set seperatly if it all works ok.
# Also make the image dirs empty if doing the cv shuffle
