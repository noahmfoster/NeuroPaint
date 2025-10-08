import torch
from accelerate import Accelerator
import numpy as np
import os
import warnings
from models.mae_with_hemisphere_embed_and_diff_dim_per_area import MAE_with_region_stitcher, NeuralStitcher_cross_att
from loader.data_loader_unbalanced_lump_short_list import *
from utils.config_utils import config_from_kwargs, update_config
from utils.utils import set_seed
from torch.optim.lr_scheduler import OneCycleLR
from trainer.make_unbalanced import make_trainer
import argparse
import pickle

warnings.simplefilter("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["WANDB_IGNORE_COPY_ERR"] = "true"


def main(include_opto, eids, with_reg):
    print(f"Include opto: {include_opto}")
    print(f"eids: {eids}")
    print(f"with_reg: {with_reg}")

    torch.cuda.empty_cache()

    #%% set arguments
    multi_gpu = True
    consistency = True
    load_previous_model = False

    base_path = '/work/hdd//bdye/jxia4/results/mae_results'
    num_train_sessions = len(eids)
    train = True

    mask_mode = 'region' # 'time' or 'region' or 'time_region'

    region_channel_num_encoder = 48 # number of region channels in encoder
    unit_embed_dim = 50
    n_layers = 5

    num_epochs = 1000
    batch_size = 16
    use_wandb = True

    #%%
    kwargs = {
        "model": f"include:/work/hdd/bdye/jxia4/code/autoencoder_mae/src/configs/mae_with_hemisphere_embed_and_diff_dim_per_area.yaml",
    }

    config = config_from_kwargs(kwargs)
    config = update_config("/work/hdd/bdye/jxia4/code/autoencoder_mae/src/configs/finetune_sessions_trainer.yaml", config)

    config['model']['encoder']['masker']['mask_mode'] = mask_mode
    config['model']['encoder']['stitcher']['n_channels_per_region'] = region_channel_num_encoder
    config['model']['encoder']['stitcher']['unit_embed_dim'] = unit_embed_dim
    config['training']['num_epochs'] = num_epochs
    config['wandb']['use'] = use_wandb
    config['wandb']['project'] = 'mae-svoboda-unbalanced'
    
    config['model']['encoder']['transformer']['n_layers'] = n_layers

    meta_data = {}

    # set accelerator
    if multi_gpu:
        print("Using multi-gpu training.")
        from accelerate.utils import DistributedDataParallelKwargs
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[kwargs]) 

        global_batch_size = batch_size * accelerator.num_processes
        config['optimizer']['lr'] = 1e-3 * global_batch_size / 256

    else:
        accelerator = Accelerator()
        
    print(f"Accelerator device: {accelerator.device}")
    rank = accelerator.process_index
    world_size = accelerator.num_processes

     
    dataloader, num_neurons, datasets, areaoi_ind, area_ind_list_list, heldout_info_list, trial_type_dict = make_loader(eids, batch_size, include_opto=include_opto, seed = config.seed, distributed=multi_gpu, rank=rank, world_size=world_size)
    set_seed(config.seed)  

    if trial_type_dict['hit_left'] != 0:
        print('warning: hit_left is not 0; consistency losss is wrong')
    if trial_type_dict['hit_right'] != 21:
        print('warning: hit_right is not 21; consistency losss is wrong')

    meta_data['area_ind_list_list'] = area_ind_list_list
    meta_data['areaoi_ind'] = areaoi_ind
    meta_data['num_sessions'] = len(eids)
    meta_data['eids'] = eids

    #load pr_max_dict.pkl
    pr_max_dict_path = '/work/hdd/bdye/jxia4/data/tables_and_infos/pr_max_dict.pkl'
    with open(pr_max_dict_path, 'rb') as f:
        pr_max_dict = pickle.load(f)

    for k, v in pr_max_dict.items():
        pr_max_dict[k] = int(v)

    meta_data['pr_max_dict'] = pr_max_dict

    trial_type_values = list(trial_type_dict.values())
    meta_data['trial_type_values'] = trial_type_values

    config = update_config(config, meta_data) # so that everything is saved in the config file

    train_dataloader = dataloader['train']
    val_dataloader = dataloader['val']
    #test_dataloader = dataloader['test']

    print('check heldout info of dataset')
    print(heldout_info_list)

    #%%
    if train:
        log_dir = os.path.join(base_path, 
                        "train",
                        "svoboda_lump_short_list_wo_trial_type_embed",
                        "with_reg_{}_2check".format(with_reg),
                        "include_opto_{}".format(include_opto),
                        "consistency_{}".format(consistency),
                        "n_layers_{}".format(n_layers),
                        "encoder_factors_{}".format(region_channel_num_encoder),
                        "unit_embed_dim_{}".format(unit_embed_dim),
                        "num_session_{}".format(num_train_sessions)
                        )
        
        os.makedirs(log_dir, exist_ok=True)

        if not torch.cuda.is_available():
            print("CUDA is not available. Using CPU.")
            exit()
        else:
            print("CUDA is available. Using GPU.")

        if config.wandb.use and accelerator.is_local_main_process and accelerator.process_index == 0:
            import wandb
            wandb.init(project=config.wandb.project,
                       dir="/work/hdd/bdye/jxia4/wandb", 
                    entity=config.wandb.entity, 
                    config=config, 
                    name="wo_trial_type_embed_with_reg_{}_2check_include_opto_{}_consistency_{}_n_layers_{}_num_session_{}_region_factors_encoder_{}".format(with_reg, include_opto, consistency, n_layers, num_train_sessions, region_channel_num_encoder)
                    )
        
        model = MAE_with_region_stitcher(config.model, **meta_data)

        if load_previous_model:
            previous_model_path = f'{base_path}/finetune/svoboda_lump_plus_10dim/include_opto_{include_opto}/consistency_{consistency}/n_layers_{n_layers}/encoder_factors_{region_channel_num_encoder}/unit_embed_dim_{unit_embed_dim}/num_session_{num_train_sessions}/{eids}/model_best.pt'
            state_dict = torch.load(previous_model_path, map_location=accelerator.device)['model']
            model.load_state_dict(state_dict)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd, eps=config.optimizer.eps)
        lr_scheduler = OneCycleLR(
                        optimizer=optimizer,
                        total_steps=config.training.num_epochs*len(train_dataloader) //config.optimizer.gradient_accumulation_steps,
                        max_lr=config.optimizer.lr,
                        pct_start=config.optimizer.warmup_pct,
                        div_factor=config.optimizer.div_factor,
                    )

        model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)

        if consistency:
            encoder_stitcher_ema = NeuralStitcher_cross_att(meta_data['eids'], meta_data['area_ind_list_list'], meta_data['areaoi_ind'], config.model.encoder.stitcher)
            encoder_stitcher_ema = accelerator.prepare(encoder_stitcher_ema)
            for param in encoder_stitcher_ema.parameters():
                param.detach_() 

        trainer_kwargs = {
            "log_dir": log_dir,
            "accelerator": accelerator,
            "lr_scheduler": lr_scheduler,
            "config": config,
            "multi_gpu": multi_gpu,
            "with_reg": with_reg,
        }
        trainer = make_trainer(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=val_dataloader,
            optimizer=optimizer,
            consistency = consistency,
            encoder_stitcher_ema = encoder_stitcher_ema if consistency else None,
            **trainer_kwargs,
            **meta_data
        )
        # check device
        print(accelerator.device)
        # train loop
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_opto", action='store_true', help='Include opto trials')
    parser.add_argument('--eids', type=int, nargs='+', help='List of EIDs')
    parser.add_argument('--with_reg', action='store_true', help='Use regularization')

    args = parser.parse_args()
    main(args.include_opto, args.eids, args.with_reg)        
        