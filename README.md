# CoRaX-Collaborative-radiology-xpert-
Enhancing Radiological Diagnosis: A Collaborative Approach Integrating AI and Human Expertise for Visual Miss Correction




<img width="1008" alt="Screenshot 2024-01-20 at 1 53 56 AM" src="https://github.com/a04101999/CoRaX-Collaborative-radiology-xpert-/assets/30754423/67034d3b-70be-49f8-abd3-f1ea5ae9547c">


# Important:  Download the Error Dataset Files here:

https://drive.google.com/drive/folders/1h9ZoITAITS_mvGjyZi8dHUpo8dMn_tHz?usp=sharing

## CoRaX contains two Trainable  modules:

1) Chexformer
2) Spatio-Temporal Abnormal Region Extractor (STARE  )

## Chexformer

Orginally Chexformer is trained on the Chexpert Dataset and we provide the pretrained chexformer model below. We also provide a sample of dataset for training and testing the Chexformer on the below link 

1) Chexformer model Pretrained model

   https://drive.google.com/file/d/1SJeXGdqveZerVSfHEFRsxVo3TPY1FoId/view?usp=sharing
   
2) Chexformer Dataset [ Train & Val ]

   https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
   

### Training Chexformer 
```
python main.py  --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataroot data/
```
### Evaluating Chexformer

```
python main.py  --batch_size 16  --lr 0.00001 --optim 'adam' --layers 3  --dataroot data/ --inference --saved_model_name=''
```
## Spatio-Temporal Abnormal Region Extractor (STARE  )

STARE module is trained on the  REFLACX and Egd-cxr. We provide the detailed discription and download link for the pre-processed dataset on the Data readme file. We provide the pre-trained model for STARE below 

1) STARE ( Pretrained model ):

   
3) Sample Training Data (Not full )
   
    - Extracted Video spatial features:  Frame features extracted by clipvit(spatial encoder )  can be downloaded below for the STARE module [ Train + Some Test samples]
      
      https://drive.google.com/file/d/1rwNMLTfh0twaSlIqu9vY93OTFb1GT1kL/view?usp=drive_link
      
    - Ground Truth Summarized  Reports: [Only Train]
  
      https://drive.google.com/file/d/1io2FGE2IC1LtLmFWLO3yIB9BAq5SNKzL/view?usp=sharing
  
    - Input Summarized  Reports:[Only Train]
  
      https://drive.google.com/file/d/116pd5J2UZWykJqFZ4NLejxtZBkAD65ve/view?usp=sharing

      
   

### Training STARE

```
python -m torch.distributed.launch --nproc_per_node 8 --use_env dvc.py --epochs=100 --lr=3e-4 --save_dir=vit --batch_size=2 --batch_size_val=2 --schedule="cosine_with_warmup"

```
## CoRaX 

To run the CoRaX on the Error dataset please use the Notebook uploaded above  ## CoRaX-Errror_correction_final_file.ipynb ## . It uses the pretrained Chexformer and STARE which is provided on the below link. Please download from there.

#### STARE

https://drive.google.com/file/d/17fPfag9rBNmHMdFrwWIIfWUX6OKbjWZ9/view?usp=sharing

#### ChexFormer

https://drive.google.com/file/d/1SJeXGdqveZerVSfHEFRsxVo3TPY1FoId/view?usp=sharing

#### Error Datasets with missing abnormalities mentioned in table-1.  All the results are calculated on these error datasets

##### Random Making based error Dataset

https://drive.google.com/drive/folders/1h9ZoITAITS_mvGjyZi8dHUpo8dMn_tHz?usp=sharing

##### Uncertainaty Making based error Dataset

https://drive.google.com/file/d/19jHWV2FQS3Uakyl1SP0GSy4H-hWragF-/view?usp=sharing

#### All the results in the paper is produced using  the below predictions by CoRaX:

##### Random Making error Dataset Predictions 

https://drive.google.com/file/d/1XK9AoiXHKegUTQPK8mZW1VGfRCjySgnR/view?usp=sharing

##### Uncertainty Making error Dataset Predictions 

https://drive.google.com/file/d/1B0CE2aDndd_sWrRNZOkp2Lk85fGOrsIq/view?usp=sharing





