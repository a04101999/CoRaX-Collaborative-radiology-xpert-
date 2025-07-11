[Under developement]

 This repository contains the technical AI development associated with a **proof-of-concept study**, accepted for publication in *Radiology: Artificial Intelligence* Journal.

The pretrained models and sample error data provided here correspond to our Accepted work in Radiology: Artificial Intelligence Journal.

🧪 This work is intended **solely for research and educational purposes**.  
🚫 It is **not for clinical use**,commercial use, diagnosis, or patient care.

We are working on extending it for future radiology education and other real time user studies. Please contact the corresponding author for collaboration.

akashcseklu123@gmail.com


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

Chexformer is trained with the five lables ( Cardiomegaly, Pleural Effusion, Lung Opacity, Atelectasis, Edema ) that are used  in CoRaX for better performence. You can use the 14 labels as well based on the need of your study



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

      
   
We are also working on the user-study with this system in future so the code for submodules STARE and CheXformer is kept for future release.

## CoRaX 

To run the CoRaX on the Error dataset please use the Notebook uploaded above  ## CoRaX-Errror_correction_final_file.ipynb ## . It uses the pretrained Chexformer and STARE which is provided on the below link. Please download from there.

#### STARE

https://drive.google.com/file/d/17fPfag9rBNmHMdFrwWIIfWUX6OKbjWZ9/view?usp=sharing

#### ChexFormer

https://drive.google.com/file/d/1SJeXGdqveZerVSfHEFRsxVo3TPY1FoId/view?usp=sharing

#### Error Datasets with missing abnormalities mentioned in table-1.  All the results are calculated on these error datasets

##### Random-Making based error Dataset

https://drive.google.com/drive/folders/1h9ZoITAITS_mvGjyZi8dHUpo8dMn_tHz?usp=sharing

##### Uncertainaty Making based error Dataset

https://drive.google.com/file/d/19jHWV2FQS3Uakyl1SP0GSy4H-hWragF-/view?usp=sharing

#### All the results in the paper is produced using  the below predictions by CoRaX:

##### Random Making error Dataset Predictions 

https://drive.google.com/file/d/1XK9AoiXHKegUTQPK8mZW1VGfRCjySgnR/view?usp=sharing

##### Uncertainty Making error Dataset Predictions 

https://drive.google.com/file/d/1B0CE2aDndd_sWrRNZOkp2Lk85fGOrsIq/view?usp=sharing

Note for real time Use-Case: There is always a lot of inter-observer variation among radiologists especially for Atelectasis, Edema and Pneumonia[ref]. Therefore the evaluation of CoRaX is very subjective. During evaluation, Please carefully evaluate the lung opacity referral since our code evaluation is based on word matching therefore it is good to also include the word like Densities, opacities, lung haziness, consolidation in the word matching for lung opacity.
Sometimes, In case of pneumonia, atlectasis and edema , Model output the Lung opacity as referral for same region of atelectasis, pneumonia edema. Consider those referral as correct since lung opacity cases can be associated with both atelectasis, consolidation , pneumonia, edema.

Ref: 'https://pubs.rsna.org/doi/full/10.1148/ryai.2019180041'

# References
Thanks to all these cool works. Our STARE submodule is inspired with the below works:

Awasthi, A., Le, N., Deng, Z., Agrawal, R., Wu, C.C. and Van Nguyen, H., 2024. Bridging human and machine intelligence: Reverse-engineering radiologist intentions for clinical trust and adoption. Computational and Structural Biotechnology Journal, 24, pp.711-723.


Awasthi, A., Ahmad, S., Le, B. and Nguyen, H., 2024, May. Decoding Radiologists’ Intentions: A Novel System for Accurate Region Identification in Chest X-Ray Image Analysis. In 2024 IEEE International Symposium on Biomedical Imaging (ISBI) (pp. 1-5). IEEE.

Yang, A., Nagrani, A., Seo, P.H., Miech, A., Pont-Tuset, J., Laptev, I., Sivic, J. and Schmid, C., 2023. Vid2seq: Large-scale pretraining of a visual language model for dense video captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10714-10726).

Lanchantin, J., Wang, T., Ordonez, V. and Qi, Y., 2021. General multi-label image classification with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 16478-16488).

