# A Discriminative Siamese Network with Softmax loss.
#### Generate Dataset

Generate two-patchs dataset randomly and classification dataset, including CASIA-WebFace trained dataset and lfw test dataset. Detailly, all network are trained and tested on the same dataset, but for Siamese Network it is two-patchs, and for Classification Network one-patch.
**Training on CASIA-WebFace, and test on lfw.**
#### Trace

1. Softmax_loss
    * CaffeNet Off-the-shelf
    * AlexNet Off-the-shelf
    * CaffeNet Fine-tuning
    * AlexNet Fine-tuning

2. Softmax_loss + Center_loss
    * CaffeNet Off-the-shelf
    * AlexNet Off-the-shelf
    * CaffeNet Fine-tuning
    * AlexNet Fine-tuning

3. Contrastive_loss on Siamese Network
    * CaffeNet Off-the-shelf
    * AlexNet Off-the-shelf    
4. 2-Channels Network
    * CaffeNet Off-the-shelf
    * AlexNet Off-the-shelf  
5. A Discriminative Siamese Network with Softmax loss
    * CaffeNet Off-the-shelf
    * AlexNet Off-the-shelf  

      

