import os

# Depends on the THRESHOLD number
DATA_SOURCE           = 'run_channels_more_300'
THRESHOLD_NAME        = '300' # or 10k
# Training method used
TRAINING_METHOD       = 'channel_sampling_then_combination'

COMMON_DLAB_PATH      = os.path.join('/dlabdata1/youtube_large/jouven/word2vec_pytorch', DATA_SOURCE, TRAINING_METHOD)
COMMON_PERSO_PATH     = os.path.join('/home/jouven/youtube_projects/word2vec_pytorch', DATA_SOURCE, TRAINING_METHOD)

DISPLAY_BATCH_LOSS    = True


if DATA_SOURCE == 'test':
    
    DISPLAY_EVERY_N_BATCH = 1
    SAVE_EVERY_N_EPOCH    = 1
    BATCH_SIZE            = 100
    NUM_EPOCHS            = 10
    
    CONTEXT               = True
    CONTEXT_SIZE          = 100
    SUBSAMPLING           = False
    SAMPLING_RATE         = 0.001
    NEGATIVE_SAMPLES      = 20 # set it to 0 if you don't want to use negative samplings

    EMBEDDING_DIM         = 10
    LR                    = 0.001

elif DATA_SOURCE == 'run_channels_more_10k':
    
    DISPLAY_EVERY_N_BATCH = 1
    SAVE_EVERY_N_EPOCH    = 1
    BATCH_SIZE            = 30000
    NUM_EPOCHS            = 1
    
    CONTEXT               = True
    CONTEXT_SIZE          = 10
    SUBSAMPLING           = False
    SAMPLING_RATE         = 0.0043
    NEGATIVE_SAMPLES      = 35 # set it to 0 if you don't want to use negative samplings  

    EMBEDDING_DIM         = 200
    LR                    = 0.01
    
elif DATA_SOURCE == 'run_channels_more_300':
    
    DISPLAY_EVERY_N_BATCH = 1
    SAVE_EVERY_N_EPOCH    = 1
    BATCH_SIZE            = 30000
    NUM_EPOCHS            = 1
    
    CONTEXT               = True
    CONTEXT_SIZE          = 100
    SUBSAMPLING           = False
    SAMPLING_RATE         = 0.001
    NEGATIVE_SAMPLES      = 35 # set it to 0 if you don't want to use negative samplings  

    EMBEDDING_DIM         = 200
    LR                    = 0.005

    
MODEL_ID = os.path.join('CONTEXT_' + str(CONTEXT) + '_' + str(CONTEXT_SIZE) + '_' + 'SUBSAMPLING_' + str(SUBSAMPLING) + '_' + str(SAMPLING_RATE)) 

MODEL_DLAB_DIR         = os.path.join(COMMON_DLAB_PATH, MODEL_ID) 
MODEL_PERSO_DIR        = os.path.join(COMMON_PERSO_PATH, MODEL_ID) 

# Path where the training data is stored
TRAINING_DATA_PATH     = os.path.join(COMMON_DLAB_PATH, 'training_data_context_'+str(CONTEXT_SIZE)+'.csv.gz')

PREPROCESSED_DATA_DIR  = os.path.join(MODEL_DLAB_DIR, 'preprocessed_data')
PREPROCESSED_DATA_PATH = os.path.join(MODEL_DLAB_DIR, PREPROCESSED_DATA_DIR, 'preprocessed_data.pickle')
EMBEDDING_PATH         = os.path.join(MODEL_DLAB_DIR, 'embedding.pth')
SUMMARY_DIR            = os.path.join(MODEL_DLAB_DIR, 'summary') 
MODEL_DIR              = os.path.join(MODEL_DLAB_DIR, 'models')