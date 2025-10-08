from accelerate import Accelerator
import os 
import numpy as np
import torch
import torch.nn as nn
from models.mae_with_region_cross_att_stitcher_simple_decoder import MAE_with_region_stitcher
from utils.config_utils import config_from_kwargs, update_config
from utils.utils import set_seed, move_batch_to_device
from loader.chaotic_rnn_loader import *
from utils.metric_utils import Poisson_fraction_deviance_explained, get_deviance_explained
#%%
def main():
    eids = list(np.arange(10, dtype=int)) 
    base_path = '/work/hdd/bdye/jxia4/results/mae_results/'
    num_train_sessions = len(eids)

    mask_mode = 'region'

    region_channel_num = 24 # number of region channels in decoder
    region_channel_num_encoder = 48 # number of region channels in encoder
    unit_embed_dim = 50
    n_layers = 5

    num_epochs = 1000
    batch_size = 16
    use_wandb = False
    consistency = True
    with_reg = True

    kwargs = {
        "model": f"include:/work/hdd/bdye/jxia4/code/autoencoder_mae/src/configs/mae_with_region_stitcher_cross_att_simple_decoder.yaml",
    }

    config = config_from_kwargs(kwargs)
    config = update_config("/work/hdd/bdye/jxia4/code/autoencoder_mae/src/configs/finetune_sessions_trainer.yaml", config)

    config['model']['encoder']['masker']['mask_mode'] = mask_mode
    config['model']['decoder']['stitcher']['n_channels_per_region'] = region_channel_num
    config['model']['encoder']['stitcher']['n_channels_per_region'] = region_channel_num_encoder
    config['model']['encoder']['stitcher']['unit_embed_dim'] = unit_embed_dim
    config['model']['encoder']['transformer']['n_layers'] = n_layers
    config['training']['num_epochs'] = num_epochs
    config['wandb']['use'] = use_wandb

    meta_data = {}

    set_seed(config.seed)   
    dataloader, num_neurons, datasets, area_ind_list_list, record_info_list = make_chaotic_rnn_loader(eids, batch_size=batch_size)
    set_seed(config.seed)
    areaoi_ind = np.array([0,1,2,3,4])
    n_area = len(areaoi_ind)

    meta_data['area_ind_list_list'] = area_ind_list_list
    meta_data['areaoi_ind'] = areaoi_ind
    meta_data['num_sessions'] = len(eids)
    meta_data['eids'] = eids

    config = update_config(config, meta_data) # so that everything is saved in the config file

    test_dataloader = dataloader['test']

    accelerator = Accelerator()

    model_path = f'{base_path}train/chaotic_rnn_g3_01_log_fr_max_3_min_neg_3/with_reg_{with_reg}/consistency_{consistency}/n_layers_{n_layers}/encoder_factors_{region_channel_num_encoder}/unit_embed_dim_{unit_embed_dim}/num_session_{num_train_sessions}/region_factors_{region_channel_num}/model_best.pt'
    model = MAE_with_region_stitcher(config.model, **meta_data)

    state_dict = torch.load(model_path)['model']
    model.load_state_dict(state_dict)
    model = accelerator.prepare(model)

    model.eval()

    #%%
    save_path = f'{base_path}eval/chaotic_rnn_g3_01_log_fr_max_3_min_neg_3/with_reg_{with_reg}/consistency_{consistency}/n_layers_{n_layers}/encoder_factors_{region_channel_num_encoder}/unit_embed_dim_{unit_embed_dim}/num_session_{num_train_sessions}/region_factors_{region_channel_num}/dfe/'
    os.makedirs(save_path, exist_ok=True)
    print(save_path)
    
    #%%
    dfe_no_mask_from_record_to_unrecord = {}
    dfe_no_mask_pred = {}
    dfe_gt = {}
    fr_pred_test_save = {}
    fr_pred_test_save_baseline = {}
    spike_test_save = {}
    

    set_seed(config.seed)  

    with torch.no_grad():
        for batch in test_dataloader:
            device = accelerator.device
            batch = move_batch_to_device(batch, device)
            B = batch['spikes_data'].size(0)
            T = batch['spikes_data'].size(1)
            eid = batch['eid'][0].item()
            print(eid)
            neuron_regions = batch['neuron_regions'][0] #(N,) area_ind_list
            area_ind_unique_tensor = neuron_regions.unique()
            R = len(area_ind_unique_tensor) #number of regions

            #don't mask any data
            mask_T = torch.zeros([B,T], dtype=torch.bool, device=device)
            mask_R = torch.zeros([B,R], dtype=torch.bool, device=device)
            ids_restore_R = None
            ids_restore_T = None
            mask = torch.zeros([B,T,R], dtype=torch.bool, device=device)

            force_mask ={'mask': mask, 
                        'mask_R': mask_R, 
                        'mask_T': mask_T, 
                        'ids_restore_R': ids_restore_R, 
                        'ids_restore_T': ids_restore_T}

            area_ind_list_full = batch['neuron_regions_full'][0] # (N_all,) 

            x, mask, mask_R, mask_T, ids_restore_R, ids_restore_T = model.encoder(batch['spikes_data'], 
                                                                                batch['spikes_timestamps'],
                                                                                area_ind_unique_tensor, 
                                                                                batch['neuron_regions'],
                                                                                batch['trial_type'], 
                                                                                mask_mode, 
                                                                                batch['eid'][0], 
                                                                                force_mask)
            
            # Decode
            factors_pred = model.stitch_decoder.Linear(x) # B x T x R_all x C

            dfe_no_mask_pred[eid] = {}
            dfe_no_mask_from_record_to_unrecord[eid] = {}
            dfe_gt[eid] = {}

            fr_pred_test_save[eid] = {}
            spike_test_save[eid] = {}
            fr_pred_test_save_baseline[eid] = {}

            n_trial_train = int(B*0.6)

            for area_ind in range(n_area):
                factors_region = factors_pred[:,:,area_ind,:]
                spikes_region = batch['spikes_data_full'][:,:,area_ind_list_full==area_ind]

                factors_region_train = factors_region[:n_trial_train]
                spikes_region_train = spikes_region[:n_trial_train]

                factors_region_test = factors_region[n_trial_train:]
                spikes_region_test = spikes_region[n_trial_train:]
                
                fr_pred_train, weight, bias, dfe_train = get_deviance_explained(factors_region_train, spikes_region_train, device, verbose=True)
                fr_pred_test = torch.exp(factors_region_test @ weight + bias[None, None, :])
                dfe_test = Poisson_fraction_deviance_explained(fr_pred_test.cpu().detach().numpy(), spikes_region_test.cpu().detach().numpy())
                
                dfe_no_mask_pred[eid][area_ind] = dfe_test
                
                #print(f'for eid {eid} region {i}, test dfe is {np.mean(dfe_test)}+/-{np.std(dfe_test)}')
                fr_pred_test_save[eid][area_ind] = fr_pred_test.cpu().detach().numpy()
                spike_test_save[eid][area_ind] = spikes_region_test.cpu().detach().numpy()

                factors_region = batch['spikes_data']
                factors_region_train = factors_region[:n_trial_train]
                factors_region_test = factors_region[n_trial_train:]

                fr_pred_train, weight, bias, dfe_train = get_deviance_explained(factors_region_train, spikes_region_train, device, verbose=True)
                fr_pred_test = torch.exp(factors_region_test @ weight + bias[None, None, :])
                fr_pred_test_save_baseline[eid][area_ind] = fr_pred_test.cpu().detach().numpy()

                dfe_test = Poisson_fraction_deviance_explained(fr_pred_test.cpu().detach().numpy(), spikes_region_test.cpu().detach().numpy())
                dfe_no_mask_from_record_to_unrecord[eid][area_ind] = dfe_test

                fr_region = batch['fr'][n_trial_train:,:,area_ind_list_full==area_ind]
                dfe_gt[eid][area_ind] = Poisson_fraction_deviance_explained(fr_region.cpu().detach().numpy(), spikes_region_test.cpu().detach().numpy())

            #break

        #break
        np.save(save_path+'dfe_gt.npy', dfe_gt)
        np.save(save_path+'mae_no_mask_fr_pred_test_save.npy', fr_pred_test_save)
        np.save(save_path+'spike_test_save.npy', spike_test_save)
        np.save(save_path+'baseline_no_mask_fr_pred_test_save.npy', fr_pred_test_save_baseline)
        np.save(save_path+'dfe_no_mask_pred.npy', dfe_no_mask_pred)
        np.save(save_path+'dfe_no_mask_from_record_to_unrecord.npy', dfe_no_mask_from_record_to_unrecord)

if __name__=="__main__":
    main()

   
#%%
