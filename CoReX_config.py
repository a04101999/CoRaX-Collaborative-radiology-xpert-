
'dataroot':'/home/cougarnet.uh.edu/aawasth3/Eye_Gaze_Research_Data_Set/images/jpg/',
'dataset':'coco',
'workers':10,
'results_dir':'results1/',
'test_known':0,




# Optimization
'optim':'adam',
'lr':0.0002,
'batch_size':32,
'test_batch_size':-1,
'grad_ac_steps':1,
'scheduler_step':1000,
'scheduler_gamma':0.1,
'epochs' :100,
'int_loss':0.0,
'aux_loss' :0.0,
'loss_type' :'bce',
'scheduler_type' :'plateau',
'loss_labels':'all',
'lr_decay':0,
'weight_decay':1e-4,
'max_samples':-1,
'max_batches':-1,
'warmup_scheduler':'',

# Model
'layers':3,
'heads':4,
'dropout':0.1,
'pos_emb':False, 
'use_lmt':True,
'freeze_backbone':False,
'no_x_features':False,



# Image Sizes
'scale_size':640,
'crop_size':576,

# Testing Models
'inference':True,
'resume':False,
'saved_model_name':'',

'overwrite':False,
'name':'',
'num_labels':14,
'epoch':1,
'train_known_labels':0,
'test_known_labels':0,
'attr_group_dict':'',

'n_groups':10


PRESAVE_DIR = "/home/cougarnet.uh.edu/aawasth3/VidChapters/"
MODEL_DIR = "/home/cougarnet.uh.edu/aawasth3/VidChapters/"
DATA_DIR = "/home/cougarnet.uh.edu/aawasth3/VidChapters/"
SSD_DIR = "/home/cougarnet.uh.edu/aawasth3/VidChapters/"
NLTK_FOLDER = "/home/cougarnet.uh.edu/aawasth3/VidChapters/"
name2folder = {
"youcook": "YouCook2",
"htm": "howto100m",
"chapters": "AllChapters",

# Dataset specific
"combine_datasets":['youcook'],


"combine_datasets_val":['youcook'],





"youcook_features_path":os.path.join(DATA_DIR, name2folder["youcook"], "clipvitl14.pth"),


"youcook_train_json_path":os.path.join(DATA_DIR, name2folder["youcook"], "train.json"),


"youcook_val_json_path":os.path.join(DATA_DIR, name2folder["youcook"], "train.json"),


"youcook_subtitles_path":os.path.join(DATA_DIR, name2folder["youcook"], "youcook2_asr_align_proc.pkl"),






"denoising":1.,
"generative":1.,
"genasr":False,
"random":False,
"mask_prob":0.25,

'mask_len':5,
"lr":3e-4,
"beta1":0.9,
"beta2":0.999,

"batch_size":1,

"batch_size_val":1,
"weight_decay":0, 
"epochs":20, 
"optimizer":"adam",

"label_smoothing":0.1,
"clip_max_norm":1., 

"schedule":'cosine_with_warmup',


"fraction_warmup_steps":0.1,

"eval_skip":1,

"print_freq":100,


# Run specific

"save_dir":"youcook_final_results",

"presave_dir":PRESAVE_DIR,
"device":"cuda",
"seed":42, 

"load":'/home/cougarnet.uh.edu/aawasth3/VidChapters/youcook_exp/best_model.pth' ,


"resume":False,


"start-epoch":0,

"eval":True,
"num_workers":3, 


"world-size":1,

"dist-url":"env://",

"model_name":"t5-base",



"bert_name":"bert-base-uncased",

"text_encoder_dropout":0.1, 

"text_decoder_dropout":0.1,

"visual_encoder_dropout":0.1, 
"max_feats":100,

"features_dim":768,


"embedding_dim":768,


"mlp_dim":2048,


"depth":12,


"heads":12,

"num_bins":100,


"use_video":True,


"use_speech":True,


"max_input_tokens":1000,



"max_output_tokens":256,



"num_beams":4,



"length_penalty":1.0,


"repetition_penalty":1.0,


"top_p":0.9,

"blip2_model_name":"pretrain_flant5xl_vitL",

"resolution":224,

"video_example":'',


"asr_example":''

