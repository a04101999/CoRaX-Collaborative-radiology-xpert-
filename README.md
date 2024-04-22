# CoRaX-Collaborative-radiology-xpert-
Enhancing Radiological Diagnosis: A Collaborative Approach Integrating AI and Human Expertise for Visual Miss Correction




<img width="1008" alt="Screenshot 2024-01-20 at 1 53 56 AM" src="https://github.com/a04101999/CoRaX-Collaborative-radiology-xpert-/assets/30754423/67034d3b-70be-49f8-abd3-f1ea5ae9547c">




## CoRaX contains two Trainable  modules:

1) Chexformer
2) Spatio-Temporal Abnormal Region Extractor (STARE  )

## Chexformer

Orginally Chexformer is trained on the Chexpert Dataset and we provide the pretrained chexformer model below. We also provide a sample of dataset for training and testing the Chexformer on the below link 

1) Chexformer model Pretrained model 
2) Chexformer Dataset [ Train & Val ]

### Training Chexformer 
```
python main.py  --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataroot data/
```
### Evaluating Chexformer

```
python main.py  --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataroot data/ --inference --saved_model_name=''
```
## Spatio-Temporal Abnormal Region Extractor (STARE  )

STARE module is trained on the combination of REFLACX and Egd-cxr. We provide the detailed discription and download link for the pre-processed dataset on the Data readme file. We provide the pre-trained model for TGP below 

1) STARE ( Pretrained model )
2) Data

### Training STARE

```
python -m torch.distributed.launch --nproc_per_node 8 --use_env dvc.py --epochs=100 --lr=3e-4 --save_dir=vit --batch_size=2 --batch_size_val=2 --schedule="cosine_with_warmup"

```
## CoRaX 

To run the CoRaX on the Error dataset please run the following command. It uses the pretrained Chexformer and TGP which is provided on the above link. Please download from there.

Error Dataset with missing abnormalities mentioned in table-1 

https://drive.google.com/file/d/1nICzyEwjQjBADP3uwfzxRfP42tfEZABe/view?usp=sharing

All the results are calculated on this error dataset

