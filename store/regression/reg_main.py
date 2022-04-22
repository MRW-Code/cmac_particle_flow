import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import glob
torch.cuda.empty_cache()
import pandas as pd

from fastai_prep_reg import RegressionFastAIPrep
from fastai.vision.all import *
from fastai.distributed import *

def empty_file(path):
    files = glob.glob(path + '/*')
    for file in files:
        os.remove(file)

empty_file('../aug_images')
empty_file('../split_test_images')

# change the second number to the appropriate split index

prep = RegressionFastAIPrep('../images', 0, 2, '../aug_images', '../split_test_images',
                            multi=True,
                            oversample=False)
prep.check_test_train_data()
df = prep.get_fastai_df()
#df.to_csv('info.csv')
print('done')

tfms = None
dls = ImageDataLoaders.from_df(df,
                               seed=None,
                               fn_col=0,
                               folder=None,
                               suff='',
                               label_col=2,
                               label_delim=None,
                               valid_col=1,
                               item_tfms=None,
                               batch_tfms=tfms,
                               bs=16,
                               val_bs=None,
                               shuffle=True,
                               device=None,
                               y_block=RegressionBlock)

learn = cnn_learner(dls, resnet18, metrics=[accuracy]).to_fp16()

learn.fine_tune(100, 0.00001, cbs=[SaveModelCallback(fname='./dump/best_cbs_100'),
                                  ReduceLROnPlateau(monitor='valid_loss',
                                                    min_delta=0.1,
                                                    patience=2)])

# for training across multiple gpus
# with learn.distrib_ctx(sync_bn=False): learn.fine_tune(100, 0.0001, cbs=[SaveModelCallback(fname='./best_cbs'),
#                                  ReduceLROnPlateau(monitor='valid_loss',
#                                                    min_delta=0.1, patience=2)])


learn.export('./dump/trained_model_100.pkl')

learn.recorder.plot_loss()
plt.savefig('./dump/training_plot_100.png')

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
plt.savefig('./dump/conf_mtrx_100.png')

