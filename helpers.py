import os
import pickle

import zstandard as zstd
import pandas as pd


'''
Using Zreader enables to read the .zst files line by line
'''
class Zreader:

    def __init__(self, file, chunk_size=16384):
        '''Init method'''
        self.fh = open(file,'rb')
        self.chunk_size = chunk_size
        self.dctx = zstd.ZstdDecompressor()
        self.reader = self.dctx.stream_reader(self.fh)
        self.buffer = ''

    def readlines(self):
        '''Generator method that creates an iterator for each line of JSON'''
        while True:
            chunk = self.reader.read(self.chunk_size).decode("utf-8", errors="ignore")
            if not chunk:
                break
            lines = (self.buffer + chunk).split("\n")

            for line in lines[:-1]:
                yield line

            self.buffer = lines[-1]

            
def check_directory(dir_1):
    
    if not os.path.exists(dir_1): 
        os.makedirs(dir_1)
        
def dict_occurent_users():
    occurent_users = {}
    with open("/dlabdata1/youtube_large/jouven/occurent_users_en.pkl",'rb') as f:
        set_occurent_users = list(pickle.load(f)[0])
    f.close()
    for user in set_occurent_users:
        occurent_users[user] = 0
    return occurent_users
        
def english_channels():
    channels = pd.read_csv("/dlabdata1/youtube_large/df_channels_en.csv.gz")
    channels_id = sorted(channels['channel'])
    # Dictionnary mapping the channel id to an integer corresponding to the row of the sparse matrix.
    dict_channel_ind = {}
    dict_ind_channel = {}
    for ind, channel_id in enumerate(channels_id):
        dict_channel_ind[channel_id] = ind
        dict_ind_channel[ind] = channel_id
    channels_id = set(channels_id)
    return dict_channel_ind, dict_ind_channel, channels_id

def video_id_to_channel_id():
    vid_to_channels = pd.read_pickle("/dlabdata1/youtube_large/id_to_channel_mapping.pickle")