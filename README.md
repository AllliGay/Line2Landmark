# Line2Landmark
Landmark Searching Based on Hand-Painted Building Lines  
Google ML Winter Camp 2020  Project 

## Background
<img src='pic/1.png' width=325>

## Tech
We use three models to realize the process of image reconstruction, object detection and image similarity comparison.

<img src='pic/2.png' width=325>

### Image Reconstruction
Model: `CycleGAN`

<img src='pic/3.png' width=325>

### Detection
Model: `Retinanet` + `FCOS`

<img src='pic/4.png' width=325>

### Image Retrieval
Model: reconstruct imgs + original imgs -> `autoencoder`

<img src='pic/5.png' width=325>

## Demo
``` python
# place models
cd webapp
python app.py # demo will run on http://127.0.0.1:5000/
```
<img src='pic/6.png' width=325>