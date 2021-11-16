# Driver-Distraction-Detection-SqueezeNet

## Purpose
This project takes the AUC Distracted Driver Dataset [1] and loads it into SqueezeNet v1.1 [2, 3] modified for a two-class classification problem. 

It can take a trained model, and then convert it from a TensorFlow model to a TensorFlow-Lite model to be ran on Jetson Nano 2GB hardware. 

SqueezeNetv1.1 is loaded with a pre-trained version which was trained on the ImageNet database to an accuracy of 80.3%. 

For further explanation of this project, please refer to reference https://dx.doi.org/10.7302/3710

## References
[1] Abouelnaga, Y., Eraqi, H. M., & Moustafa, M. N. (2017). Real-time Distracted Driver
Posture Classification. Nips. http://arxiv.org/abs/1706.09498

[2] Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., & Keutzer, K. (2016).
SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size.
1–13. http://arxiv.org/abs/1602.07360.

[3] Forresti (2018) SqueezeNet [SqueezeNet_v1.1] https://github.com/forresti/SqueezeNet
