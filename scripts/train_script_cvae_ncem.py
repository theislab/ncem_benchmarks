import pickle
import sys
import time
import tensorflow as tf

import ncem
from ncem_benchmarks import HyperparameterContainer, ConditionalHyperparameterContainer

print(tf.__version__)

# Set global variables.
print("sys.argv", sys.argv)

# manual inputs
data_set = sys.argv[1].lower()
optimizer = sys.argv[2].lower()
cond_type = sys.argv[3].lower()
domain_type = sys.argv[4].lower()

learning_rate_keys = sys.argv[5]
latent_dim_keys = sys.argv[6]
dropout_rate_key = sys.argv[7]
l1_key = sys.argv[8]
l2_keys = sys.argv[9]

encoder_intermediate_dim_key = sys.argv[10]
encoder_depth_key = sys.argv[11]
decoder_intermediate_dim_key = sys.argv[12]
decoder_depth_key = sys.argv[13]

# conditional
cond_depth_key = sys.argv[14]
cond_dim_key = sys.argv[15]
cond_dropout_rate_key = sys.argv[16]
cond_l2_req_key = sys.argv[17]

# other
batch_size_key = sys.argv[18]
radius_key = sys.argv[19]
n_eval_nodes_keys = sys.argv[20]
use_type_cond = bool(int(sys.argv[21]))
pretrain_decoder = bool(int(sys.argv[22]))
aggressive = bool(int(sys.argv[23]))

model_class = sys.argv[24].lower()
gs_id = sys.argv[25].lower()
data_path_base = sys.argv[26]
out_path = sys.argv[27]

split_mode = sys.argv[28].lower()

if data_set == 'zhang':
    data_path = data_path_base + '/zhang/'
    use_domain = True
    merge_node_types_predefined = True
    covar_selection = []
    target_cell_types = ["L23_IT"]
    radius_dict = {
        "0": 0,
        "1": 10,
        "2": 25,
        "3": 50,
        "4": 100,
        "5": 250,
        "6": 500,
        "7": 1000,
        "8": 6000,
    }
    intermediate_dim_dict = {
        "1": 4,
        "2": 8,
        "3": 16,
        "4": 32,
        "5": 64,
        "6": 128
    }
    latent_dim_dict = {
        "1": 12,
        "2": 24,
        "3": 48
    }
    log_transform = False
    scale_node_size = False
    output_layer = "gaussian_const_disp"
    pre_warm_up = 0
    max_beta = 1.
    beta = 1.
    if not aggressive and not pretrain_decoder:
        pre_warm_up = 0
        max_beta = 1.
        beta = 0.02
elif data_set == 'jarosch':
    data_path = data_path_base + '/jarosch/'
    use_domain = True
    merge_node_types_predefined = True
    covar_selection = []
    target_cell_types = ["muscular cells", "Lamina propria cells"]
    radius_dict = {
        "0": 0,
        "1": 10,
        "2": 20,
        "3": 40,
        "4": 80,
        "5": 200,
        "6": 800,
        "7": 5000
    }
    intermediate_dim_dict = {
        "1": 4,
        "2": 8,
        "3": 16,
        "4": 32,
        "5": 64,
        "6": 128
    }
    latent_dim_dict = {
        "1": 4,
        "2": 12,
        "3": 24
    }
    log_transform = True
    scale_node_size = False
    output_layer = "gaussian"
    pre_warm_up = 0
    max_beta = 1.
    beta = 1.
    if not aggressive and not pretrain_decoder:
        pre_warm_up = 0
        max_beta = 1.
        beta = 0.02
else:
    raise ValueError('data_origin not recognized')

# model and training
ncv = 3
epochs = 2000
max_steps_per_epoch = 20
patience = 100
lr_schedule_min_lr = 1e-10
lr_schedule_factor = 0.5
lr_schedule_patience = 50
val_bs = 16
max_val_steps_per_epoch = 10
epochs_warmup = int(100)  # we will train for 100 epochs without early stopping
if not aggressive and not pretrain_decoder:
    epochs_warmup = int(max_beta // beta - 1 + pre_warm_up)

feature_space_id = "standard"
cond_feature_space_id = "type"

use_covar_node_label = True
use_covar_node_position = False
use_covar_graph_covar = False

cond_activation = 'relu'
cond_use_bias = True

use_batch_norm = True
transform_input = True

hpcontainer = HyperparameterContainer()
cond_hpcontainer = ConditionalHyperparameterContainer()

for ld in latent_dim_keys.split("+"):
    for learning_rate_key in learning_rate_keys.split("+"):
        for l2_key in l2_keys.split("+"):
            for n_key in n_eval_nodes_keys.split("+"):
                # Set ID of output:
                model_id_base = f"{gs_id}_{optimizer}_lr{str(learning_rate_key)}" \
                           f"_bs{str(batch_size_key)}_md{str(radius_key)}_n{str(n_key)}" \
                           f"_fs{str(feature_space_id)}_l2{str(l2_key)}_l1{str(l1_key)}"
                model_id_cvae = f"_ldi{str(ld)}_ei{str(encoder_intermediate_dim_key)}_" \
                                f"di{str(decoder_intermediate_dim_key)}_ede{str(encoder_depth_key)}_" \
                                f"dde{str(decoder_depth_key)}_dr{str(dropout_rate_key)}_" \
                                f"agg{str(aggressive)}_predec{str(pretrain_decoder)}"
                model_id_cond = f"_COND_cde{str(cond_depth_key)}_cb{str(cond_use_bias)}" \
                                f"_cdi{str(cond_dim_key)}_cdr{str(cond_dropout_rate_key)}_cl2{str(cond_l2_req_key)}"
                model_id = model_id_base + model_id_cvae + model_id_cond
                run_params = {
                    'model_class': model_class,
                    'gs_id': gs_id,
                    'model_id': model_id,
                    'merge_node_types_predefined': merge_node_types_predefined,

                    'data_set': data_set,
                    'radius': radius_dict[radius_key],
                    'graph_covar_selection': covar_selection,
                    'node_label_space_id': cond_feature_space_id,
                    'node_feature_space_id': feature_space_id,
                    'use_covar_node_position': use_covar_node_position,
                    'use_covar_node_label': use_covar_node_label,
                    'use_covar_graph_covar': use_covar_graph_covar,

                    'optimizer': optimizer,
                    'learning_rate': hpcontainer.learning_rate[learning_rate_key],
                    'l2_coef': hpcontainer.l2_coef[l2_key],
                    'l1_coef': hpcontainer.l1_coef[l1_key],

                    'use_domain': use_domain,
                    'domain_type': domain_type,
                    'scale_node_size': scale_node_size,
                    'output_layer': output_layer,
                    'log_transform': log_transform,

                    'epochs': epochs,
                    'batch_size': hpcontainer.batch_size[batch_size_key]
                }
                kwargs_estim_init = {
                    "cond_type": cond_type,
                    "use_type_cond": use_type_cond,
                    "log_transform": log_transform,
                }
                kwargs_model_init = {
                    "optimizer": optimizer,
                    'learning_rate': hpcontainer.learning_rate[learning_rate_key],
                    'latent_dim': latent_dim_dict[ld],
                    'dropout_rate': hpcontainer.dropout[dropout_rate_key],
                    'l2_coef': hpcontainer.l2_coef[l2_key],
                    'l1_coef': hpcontainer.l1_coef[l1_key],

                    "intermediate_dim_enc": intermediate_dim_dict[encoder_intermediate_dim_key],
                    "depth_enc": hpcontainer.depth[encoder_depth_key],
                    "intermediate_dim_dec": intermediate_dim_dict[decoder_intermediate_dim_key],
                    "depth_dec": hpcontainer.depth[decoder_depth_key],

                    "cond_depth": cond_hpcontainer.conditional_depth[cond_depth_key],
                    "cond_dim": cond_hpcontainer.conditional_dimension[cond_dim_key],
                    "cond_dropout_rate": cond_hpcontainer.conditional_dropout[cond_dropout_rate_key],
                    "cond_activation": cond_activation,
                    "cond_l2_reg": cond_hpcontainer.conditional_l2_dict[cond_l2_req_key],
                    "cond_use_bias": cond_use_bias,

                    "n_eval_nodes_per_graph": hpcontainer.n_eval_nodes[n_key],

                    "use_domain": use_domain,
                    "use_batch_norm": use_batch_norm,
                    "scale_node_size": scale_node_size,
                    "transform_input": transform_input,

                    "beta": beta,
                    "max_beta": max_beta,
                    "pre_warm_up": pre_warm_up,
                    "output_layer": output_layer,
                }
                kwargs_train = {}
                run_params.update(kwargs_estim_init)
                run_params.update(kwargs_model_init)
                run_params.update(kwargs_train)

                # overall runparameters
                fn_out = out_path + "/results/" + model_id
                with open(fn_out + '_runparams.pickle', 'wb') as f:
                    pickle.dump(obj=run_params, file=f)

                for i in range(ncv):
                    t0 = time.time()
                    model_id_cv = model_id + "_cv" + str(i)
                    fn_tensorboard_cv = None  # out_path + "/logs/" + model_id_cv
                    fn_out_cv = out_path + "/results/" + model_id_cv
                    trainer = ncem.train.TrainModelCVAEncem()
                    trainer.init_estim(**kwargs_estim_init)
                    trainer.estimator.get_data(
                        data_origin=data_set,
                        data_path=data_path,
                        radius=radius_dict[radius_key],
                        graph_covar_selection=covar_selection,
                        node_label_space_id=cond_feature_space_id,
                        node_feature_space_id=feature_space_id,
                        use_covar_node_position=use_covar_node_position,
                        use_covar_node_label=use_covar_node_label,
                        use_covar_graph_covar=use_covar_graph_covar,
                        domain_type=domain_type,
                    )
                    trainer.estimator.split_data_node(
                        validation_split=0.1,
                        test_split=0.1,
                        seed=i
                    )
                    if hpcontainer.batch_size[batch_size_key] is None:
                        bs = len(list(trainer.estimator.complete_img_keys))
                        shuffle_buffer_size = None
                    else:
                        bs = hpcontainer.batch_size[batch_size_key]
                        shuffle_buffer_size = int(100)
                    trainer.estimator.init_model(
                        **kwargs_model_init
                    )
                    trainer.estimator.train(
                        epochs=epochs,
                        epochs_warmup=epochs_warmup,
                        batch_size=bs,
                        log_dir=fn_tensorboard_cv,
                        max_steps_per_epoch=max_steps_per_epoch,
                        validation_batch_size=val_bs,
                        max_validation_steps=max_val_steps_per_epoch,
                        patience=patience,
                        lr_schedule_min_lr=lr_schedule_min_lr,
                        lr_schedule_factor=lr_schedule_factor,
                        lr_schedule_patience=lr_schedule_patience,
                        monitor_partition="val",
                        monitor_metric="loss",
                        shuffle_buffer_size=shuffle_buffer_size,
                        early_stopping=True,
                        reduce_lr_plateau=True,
                        **kwargs_train
                    )
                    trainer.save(fn=fn_out_cv, save_weights=True)
                    trainer.save_time(fn=fn_out_cv, duration=time.time() - t0)
