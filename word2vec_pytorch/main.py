from __future__ import print_function
from tqdm import tqdm
# from tqdm import tqdm_gui
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import sys, os, pickle, shutil

import torch
import torch.optim as optim
import torch.nn as nn

# it is a little tricky on run SummaryWriter by installing a suitable version of pytorch. so 
# if you are able to import SummaryWriter from torch.utils.tensorboard, this script will record summaries. 
# Otherwise it will not.
try:
    from torch.utils.tensorboard import SummaryWriter
    write_summary = True
except:
    write_summary = False

from model import *
from utils_modified import count_parameters, q
from dataset2 import *
from  config import *

scriptpath = "../"
sys.path.append(os.path.abspath(scriptpath))
from helpers.helpers import *

if DATA_SOURCE == 'test':
    if os.path.exists(MODEL_ID):
        shutil.rmtree(MODEL_ID)

check_directory(MODEL_ID)
check_directory(MODEL_DIR)

# SUMMARY_DIR is the path of the directory where the tensorboard SummaryWriter files are written
if write_summary:
    if os.path.exists(SUMMARY_DIR):
        # the directory is removed, if it already exists
        shutil.rmtree(SUMMARY_DIR)

    writer = SummaryWriter(SUMMARY_DIR) # this command automatically creates the directory at SUMMARY_DIR
    summary_counter = 0
    
    
# Make training data
vocab_occ = load_data(CONTEXT, CONTEXT_SIZE, SUBSAMPLING, SAMPLING_RATE)


print('len(vocab): ', len(vocab_occ))


# make noise distribution to sample negative examples from the words (channels)
word_freqs = np.array(vocab_occ)
unigram_dist = word_freqs/sum(word_freqs)
noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))


losses = []

model = Word2Vec_neg_sampling(EMBEDDING_DIM, len(vocab_occ), DEVICE, noise_dist, NEGATIVE_SAMPLES).to(DEVICE)
print('\nWe have {} Million trainable parameters here in the model'.format(count_parameters(model)))

optimizer = optim.Adam(model.parameters(), lr = LR)

for epoch in tqdm(range(NUM_EPOCHS)):
    print('\n===== EPOCH {}/{} ====='.format(epoch + 1, NUM_EPOCHS))
    
    # model.train()
    for chunk in pd.read_csv(TRAINING_DATA_PATH, compression='gzip', chunksize = BATCH_SIZE):
        #print('batch# ' + str(batch_idx+1).zfill(len(str(len(train_loader)))) + '/' + str(len(train_loader)), end = '\r')
        model.train()
        x_batch = torch.tensor(list(chunk['0']), dtype = torch.long).to(DEVICE)
        y_batch = torch.tensor(list(chunk['1']), dtype = torch.long).to(DEVICE)

        optimizer.zero_grad()
        loss = model(x_batch, y_batch)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        
    # write embeddings every SAVE_EVERY_N_EPOCH epoch
    if epoch % SAVE_EVERY_N_EPOCH == 0:
        if write_summary:
            writer.add_embedding(model.embeddings_input.weight.data, metadata=[k for k in range(len(vocab_occ))], global_step=epoch)

        torch.save({'model_state_dict': model.state_dict(),
                    'losses': losses,
                    'embedding': model.embeddings_input.weight.data
                    },
                    '{}/model{}.pth'.format(MODEL_DIR, epoch))

plt.figure(figsize = (50, 50))
plt.xlabel("batches")
plt.ylabel("batch_loss")
plt.title("loss vs #batch")

plt.plot(losses)
plt.savefig('losses.png')
plt.show()


EMBEDDINGS = model.embeddings_input.weight.data
# Save embedding
torch.save(EMBEDDINGS, EMBEDDING_PATH)
                                    
print('EMBEDDINGS.shape: ', EMBEDDINGS.shape)
