# CoRaX-Collaborative-radiology-xpert-
Enhancing Radiological Diagnosis: A Collaborative Approach Integrating AI and Human Expertise for Visual Miss Correction
<img width="674" alt="Screenshot 2024-01-20 at 12 24 57 AM" src="https://github.com/a04101999/CoRaX-Collaborative-radiology-xpert-/assets/30754423/b86db475-3946-4b5c-8baf-5900a0f25380">


## CoRaX contains two Trainable  modules:

1) Chexformer
2) Temporal Grounding Predictor ( TGP )

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
##Temporal Grounding Predictor ( TGP )
TGP module is trained on the combination of REFLACX and Egd-cxr. We provide the detailed discription and download link for the pre-processed dataset on the Data readme file. We provide the pre-trained model for TGP below 

1) TGP ( Pretrained model )
2) Data

### Training TGP

```
python -m torch.distributed.launch --nproc_per_node 8 --use_env dvc.py --epochs=100 --lr=3e-4 --save_dir=vit --batch_size=2 --batch_size_val=2 --schedule="cosine_with_warmup"

```
