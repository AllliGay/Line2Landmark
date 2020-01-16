# Line2Landmark
Landmark Searching Based on Hand-Painted Building Lines  
Google ML Winter Camp 2020  Project 

## Background
![1.png](pic/1.png)

## Tech
We use three models to realize the process of image reconstruction, object detection and image similarity comparison.

![2.png](pic/2.png)

### Image Reconstruction
Model: `CycleGAN`

![3.png](pic/3.png)

### Detection
Model: `Retinanet` + `FCOS`

![4.png](pic/4.png)

### Image Retrieval
Model: reconstruct imgs + original imgs -> `autoencoder`

![5.png](pic/5.png)

## Demo
![6.png](pic/6.png)