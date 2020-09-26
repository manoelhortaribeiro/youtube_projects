import zstandard as zstd
import pandas as pd
import networkx as nx
import json
import queue
import time
import pickle
import numpy as np
import gzip


channelcrawler = pd.read_csv("/dlabdata1/youtube_large/channelcrawler.csv")
channelcrawler['channel_id'] = channelcrawler['link'].str.split('/').str[-1]

# The set of channels contained in channelcrawler
set_channelcrawler = set(channelcrawler['channel_id'])

# Dictionnary mapping the video_id to the channel_id
vid_to_channels = pd.read_pickle("/dlabdata1/youtube_large/id_to_channel_mapping.pickle")

# The set of channels contained in the comments dataset
set_channels = set(vid_to_channels.values())

set_videos = set(vid_to_channels.keys())

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

def add_edge(graph_dict, user_edge):
    if graph_dict.get(user_edge) is None:
        graph_dict[user_edge] = 1
    else:
        graph_dict[user_edge] += 1


graph_dict = {}
# Adjust chunk_size as necessary -- defaults to 16,384 if not specific
reader = Zreader("/dlabdata1/youtube_large/youtube_comments.ndjson.zst", chunk_size=16384)

idx = 1
first_user = True
user_edge = queue.Queue(maxsize=0) # queue corresponding to the an edge

user = 'author_id'
begin_time = time.time()

# Read each line from the reader
for line in reader.readlines():
    line_split = line.replace('"', '').split(',')
    if len(line_split) == 9:
        if idx == 1:
            print(line_split)

        else:
            if line_split[2] in set_videos:
                corr_channel = vid_to_channels[line_split[2]]
                if corr_channel in set_channelcrawler:
                    if first_user:
                        user = line_split[0]
                        first_user = False
                    if line_split[0] == user:
                        user_edge.put(corr_channel)

                        if len(user_edge.queue) == 2:
                            add_edge(graph_dict, str(tuple(user_edge.queue)))
                            #print(user_edge.queue)
                        elif len(user_edge.queue) == 3:
                            user_edge.get()
                            add_edge(graph_dict, str(tuple(user_edge.queue)))
                            #print(user_edge.queue)
                    else:
                        user_edge = queue.Queue(maxsize=0)
                        user_edge.put(corr_channel)
                        user = line_split[0]
    idx += 1
    if idx % 100000000 == 0:
        print('line number: ' + str(idx) + ' time: ' + str(time.time() - begin_time))
        begin_time = time.time()

outfilename = '../../../dlabdata1/youtube_large/jouven/simple_graph_set.json.gz'
output = gzip.open(outfilename, 'w')
output.write((str(graph_dict)).encode('utf-8'))
output.close()
