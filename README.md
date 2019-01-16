# face-video-preprocessing
For loading the data about frames:

```
./load.sh 
```

For rest of the processing (Assuming 8 gpu, and 5 workers per gpu). I use 5 per-gpu for TitanX, maybe this can be increased.
```
python read.py --workers 40 --device_ids 0,1,2,3,4,5,6,7 
```
