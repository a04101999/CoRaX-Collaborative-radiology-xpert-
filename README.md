# CoRaX-Collaborative-radiology-xpert-
Enhancing Radiological Diagnosis: A Collaborative Approach Integrating AI and Human Expertise for Visual Miss Correction


## CoRaX contains two Trainable  modules:

1) Chexformer
2) Temporal Grounding Predictor ( TGP )

## Chexformer

Orginally Chexformer is trained on the Chexpert Dataset and we provide the pretrained chexformer model below. We also provide a sample of dataset for training and testing the Chexformer on the below link 

Chexformer model Pretrained model 
Chexformer Dataset [ Train & Val ]

### Training Chexformer 

python main.py  --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataroot data/


   
