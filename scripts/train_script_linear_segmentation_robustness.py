import pickle
import sys
import time
import tensorflow as tf

import ncem
from ncem_benchmarks import HyperparameterContainer

print(tf.__version__)

# Set global variables.
print("sys.argv", sys.argv)

data_set = sys.argv[1].lower()
optimizer = sys.argv[2].lower()
domain_type = sys.argv[3].lower()
learning_rate_keys = sys.argv[4]
l1_key = sys.argv[5]
l2_keys = sys.argv[6]

batch_size_key = sys.argv[7]
radius_key = sys.argv[8]
n_eval_nodes_keys = sys.argv[9]

model_class = sys.argv[10].lower()
gs_id = sys.argv[11].lower()
data_path_base = sys.argv[12]
out_path = sys.argv[13]

if data_set == 'jarosch':
    data_path = data_path_base + '/jarosch/'
    use_domain = True
    merge_node_types_predefined = True
    covar_selection = []
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
    n_rings_dict = {
        "0": 1
    }
    n_rings_key = "0"
    log_transform = True
    scale_node_size = False
    output_layer = 'linear'
    segmentation_robustness_node_fraction = [0.1, 0.5]
    segmentation_robustness_overflow_fraction = [0., 0.05, 0.1, 0.5]
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
epochs_warmup = 0

feature_space_id = "standard"
cond_feature_space_id = "type"

use_covar_node_label = False if model_class in ["interactions", "interactions_baseline"] else True
use_covar_node_position = False
use_covar_graph_covar = False

hpcontainer = HyperparameterContainer()

for learning_rate_key in learning_rate_keys.split("+"):
    for l2_key in l2_keys.split("+"):
        for n_key in n_eval_nodes_keys.split("+"):
            for nf in segmentation_robustness_node_fraction:
                for of in segmentation_robustness_overflow_fraction:
                    # Set ID of output:
                    model_id_base = f"{gs_id}_{optimizer}_lr{str(learning_rate_key)}" \
                               f"_bs{str(batch_size_key)}_md{str(radius_key)}_ri{str(n_rings_key)}_n{str(n_key)}" \
                               f"_fs{str(feature_space_id)}_l2{str(l2_key)}_l1{str(l1_key)}_senf{str(nf)}_seof{str(of)}"
                    model_id = model_id_base
                    run_params = {
                        'model_class': model_class,
                        'gs_id': gs_id,
                        'model_id': model_id,
                        'merge_node_types_predefined': merge_node_types_predefined,

                        'data_set': data_set,
                        'radius': radius_dict[radius_key],
                        'n_rings': n_rings_dict[n_rings_key],
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

                        'segmentation_robustness_node_fraction': nf,
                        'segmentation_robustness_overflow_fraction': of
                    }
                    kwargs_estim_init = {
                        "log_transform": log_transform,
                    }
                    kwargs_train = {}
                    if model_class == "linear_baseline":
                        kwargs_model_init = {
                            "use_source_type": False
                        }
                    elif model_class == "linear":
                        kwargs_model_init = {
                            "use_source_type": True
                        }
                    elif model_class == "interactions_baseline":
                        kwargs_model_init = {
                            "use_interactions": False
                        }
                    elif model_class == "interactions":
                        kwargs_model_init = {
                            "use_interactions": True
                        }
                    else:
                        raise ValueError("model_class %s not recognized" % model_class)
                    kwargs_model_init.update({
                        "optimizer": optimizer,
                        'learning_rate': hpcontainer.learning_rate[learning_rate_key],
                        'l2_coef': hpcontainer.l2_coef[l2_key],
                        'l1_coef': hpcontainer.l1_coef[l1_key],

                        "n_eval_nodes_per_graph": hpcontainer.n_eval_nodes[n_key],

                        "use_domain": use_domain,
                        "scale_node_size": scale_node_size,
                        "output_layer": output_layer,
                    })
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
                        if model_class in ["interactions", "interactions_baseline"]:
                            trainer = ncem.train.TrainModelInteractions()
                        elif model_class in ["linear", "linear_baseline"]:
                            trainer = ncem.train.TrainModelLinear()
                        else:
                            raise ValueError("model_class %s not recognized" % model_class)
                        trainer.init_estim(**kwargs_estim_init)
                        trainer.estimator.get_data(
                            data_origin=data_set,
                            data_path=data_path,
                            radius=radius_dict[radius_key],
                            n_rings=n_rings_dict[n_rings_key],
                            graph_covar_selection=covar_selection,
                            node_label_space_id=cond_feature_space_id,
                            node_feature_space_id=feature_space_id,
                            use_covar_node_position=use_covar_node_position,
                            use_covar_node_label=use_covar_node_label,
                            use_covar_graph_covar=use_covar_graph_covar,
                            domain_type=domain_type,
                            robustness=None,
                            n_top_genes=None,
                            segmentation_robustness=[nf, of]
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
