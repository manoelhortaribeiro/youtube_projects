# Repo for YouTube-related projects

Data is located in `/dlabdata1/youtube_large/`. Quick description:

- `youtube_large/_tmp`: Ignore this folder (atleast for now :)!
- `channelcrawler.csv`: List with all channels collected;
- `rankings.csv`: Rankings of all channels collected;
- `youtube_comments.nd.json`: File with 10B comments;
- `yt_metadata_all`: File with metadata for 64M videos;

- Important: the reading scripts require `zstandard==0.11.0`, newer versions break it somehow! See `requirements.txt`;