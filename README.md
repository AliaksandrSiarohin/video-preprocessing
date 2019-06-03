# Preprocessing vox-celeb dataset
1) Load vox-celeb1(vox-celeb2) annotations:

```
wget www.robots.ox.ac.uk/~vgg/data/voxceleb/data/vox1_test_txt.zip
unzip vox1_test_txt.zip

wget www.robots.ox.ac.uk/~vgg/data/voxceleb/data/vox1_dev_txt.zip
unzip vox1_dev_txt.zip
```

```
wget www.robots.ox.ac.uk/~vgg/data/voxceleb/data/vox2_test_txt.zip
unzip vox2_test_txt.zip

wget www.robots.ox.ac.uk/~vgg/data/voxceleb/data/vox2_dev_txt.zip
unzip vox2_dev_txt.zip
```


2) Load youtube-dl:
```
wget https://yt-dl.org/downloads/latest/youtube-dl -O youtube-dl
chmod a+rx youtube-dl
```

3) Install face-alignment library:

```
git clone https://github.com/1adrianb/face-alignment
cd face-alignment
pip install -r requirements.txt
python setup.py install
```
4) Install ffmpeg

```
sudo apt-get install ffmpeg
```

5) Run preprocessing (assuming 8 gpu, and 5 workers per gpu).
```
python crop_vox.py --workers 40 --device_ids 0,1,2,3,4,5,6,7 
```
