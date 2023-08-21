# PITT
Reproduction and further exploration work of physic informed token transformer

# 2023/8/21 milestone content:
+ standard PITT content
+ fusion training
+ loss comparison of fusion and other models on three 1D datasets

# how to evaluate fusion and other trained models?
1. download 1D data files.
2. run train_pitt.py for heat, bergers and KdV respectively by changing the config iof pitt_config.yaml in /configs.
3. run train_pitt_1d_fusion.py to get fusion model.
   + trained models are in /1D_results for fusion and heat already.
5. run compare_models_1D_num_sample.py to evaluate models(only fusion and heat corrently).
