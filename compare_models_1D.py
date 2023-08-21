import h5py
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torch import nn

from models.pitt import PhysicsInformedTokenTransformer
from train_pitt_1d_fusion import get_neural_operator

from torch.utils.data import DataLoader, ConcatDataset

from utils import TransformerOperatorDataset

device = 'cuda' if (torch.cuda.is_available()) else 'cpu'


def get_data(f, config):
    train_data = TransformerOperatorDataset(f, config['flnm'],
                                            split="train",
                                            initial_step=config['initial_step'],
                                            reduced_resolution=config['reduced_resolution'],
                                            reduced_resolution_t=config['reduced_resolution_t'],
                                            reduced_batch=config['reduced_batch'],
                                            saved_folder=config['base_path'],
                                            return_text=config['return_text'],
                                            num_t=config['num_t'],
                                            num_x=config['num_x'],
                                            sim_time=config['sim_time'],
                                            num_samples=config['num_samples'],
                                            train_style=config['train_style'],
                                            rollout_length=config['rollout_length'],
                                            seed=config['seed'],
                                            val_ratio=0.2,
                                            )
    train_data.data = train_data.data.to(device)
    train_data.grid = train_data.grid.to(device)
    val_data = TransformerOperatorDataset(f, config['flnm'],
                                          split="val",
                                          initial_step=config['initial_step'],
                                          reduced_resolution=config['reduced_resolution'],
                                          reduced_resolution_t=config['reduced_resolution_t'],
                                          reduced_batch=config['reduced_batch'],
                                          saved_folder=config['base_path'],
                                          return_text=config['return_text'],
                                          num_t=config['num_t'],
                                          num_x=config['num_x'],
                                          sim_time=config['sim_time'],
                                          num_samples=config['num_samples'],
                                          train_style=config['train_style'],
                                          rollout_length=config['rollout_length'],
                                          seed=config['seed'],
                                          val_ratio=0.2,
                                          )
    val_data.data = val_data.data.to(device)
    val_data.grid = val_data.grid.to(device)
    test_data = TransformerOperatorDataset(f, config['flnm'],
                                           split="test",
                                           initial_step=config['initial_step'],
                                           reduced_resolution=config['reduced_resolution'],
                                           reduced_resolution_t=config['reduced_resolution_t'],
                                           reduced_batch=config['reduced_batch'],
                                           saved_folder=config['base_path'],
                                           return_text=config['return_text'],
                                           num_t=config['num_t'],
                                           num_x=config['num_x'],
                                           sim_time=config['sim_time'],
                                           num_samples=config['num_samples'],
                                           train_style=config['train_style'],
                                           rollout_length=config['rollout_length'],
                                           seed=config['seed'],
                                           val_ratio=0.2,
                                           )
    test_data.data = test_data.data.to(device)
    test_data.grid = test_data.grid.to(device)
    return train_data, val_data, test_data


if __name__ == '__main__':

    # x = np.load("1D_results/pitt_oformer_Fusion_varied_next_step_novel/test_vals_0.npy")

    fusion_model_path = '1D_results/pitt_oformer_Fusion_varied_next_step_novel/Fusion_pitt_0.pt'
    heat_model_path = '1D_results/pitt_oformer_Heat_varied_next_step_novel/Heat_pitt_0.pt'
    burgers_model_path = '1D_results/pitt_oformer_Burgers_varied_next_step_novel/Burgers_pitt_0.pt'
    kdv_model_path = '1D_results/pitt_oformer_KdV_varied_next_step_novel/KdV_pitt_0.pt'

    with open("./configs/pitt_config_fusion.yaml", 'r') as stream:
        train_args = yaml.safe_load(stream)
    config = train_args['args']

    neural_operator = get_neural_operator(config['neural_operator'], config)

    fusion_model = PhysicsInformedTokenTransformer(500, config['hidden'], config['layers'], config['heads'],
                                                   config['num_x'], dropout=config['dropout'],
                                                   neural_operator=neural_operator).to(device=device)
    fusion_dict = torch.load(fusion_model_path)
    fusion_model.load_state_dict(fusion_dict['model_state_dict'])

    neural_operator = get_neural_operator(config['neural_operator'], config)

    heat_model = PhysicsInformedTokenTransformer(500, config['hidden'], config['layers'], config['heads'],
                                                 config['num_x'], dropout=config['dropout'],
                                                 neural_operator=neural_operator).to(device=device)
    heat_dict = torch.load(heat_model_path)
    heat_model.load_state_dict(heat_dict['model_state_dict'])

    neural_operator = get_neural_operator(config['neural_operator'], config)

    burgers_model = PhysicsInformedTokenTransformer(500, config['hidden'], config['layers'], config['heads'],
                                                    config['num_x'], dropout=config['dropout'],
                                                    neural_operator=neural_operator).to(device=device)
    burgers_dict = torch.load(heat_model_path)
    burgers_model.load_state_dict(burgers_dict['model_state_dict'])

    neural_operator = get_neural_operator(config['neural_operator'], config)

    kdv_model = PhysicsInformedTokenTransformer(500, config['hidden'], config['layers'], config['heads'],
                                                config['num_x'], dropout=config['dropout'],
                                                neural_operator=neural_operator).to(device=device)
    kdv_dict = torch.load(kdv_model_path)
    kdv_model.load_state_dict(kdv_dict['model_state_dict'])

    heat_config = {
        'flnm': "Heat",
        'data_name': 'varied_heat_10000.h5',
        'initial_step': 10,
        'reduced_resolution': 1,
        'reduced_resolution_t': 1,
        'reduced_batch': 1,
        'base_path': './pde_data/',
        'return_text': True,
        'num_t': 100,
        'num_x': 100,
        'sim_time': 1000,
        'num_samples': 10,
        'train_style': 'next_step',
        'rollout_length': 1,
        'seed': 0,

        'batch_size': 128,
        'num_workers': 0,

    }
    KdV_config = {
        'flnm': "KdV",
        'data_name': 'varied_kdv_2500.h5',
        'initial_step': 10,
        'reduced_resolution': 1,
        'reduced_resolution_t': 1,
        'reduced_batch': 1,
        'base_path': './pde_data/',
        'return_text': True,
        'num_t': 100,
        'num_x': 100,
        'sim_time': 1000,
        'num_samples': 10,
        'train_style': 'next_step',
        'rollout_length': 1,
        'seed': 0,

        'batch_size': 128,
        'num_workers': 0,

    }
    burgers_config = {
        'flnm': "Burgers",
        'data_name': 'varied_burgers_2500.h5',
        'initial_step': 10,
        'reduced_resolution': 1,
        'reduced_resolution_t': 1,
        'reduced_batch': 1,
        'base_path': './pde_data/',
        'return_text': True,
        'num_t': 100,
        'num_x': 100,
        'sim_time': 1000,
        'num_samples': 10,
        'train_style': 'next_step',
        'rollout_length': 1,
        'seed': 0,

        'batch_size': 128,
        'num_workers': 0,

    }
    heat_f = h5py.File("{}{}".format(heat_config['base_path'], heat_config['data_name']), 'r')
    KdV_f = h5py.File("{}{}".format(KdV_config['base_path'], KdV_config['data_name']), 'r')
    burgers_f = h5py.File("{}{}".format(burgers_config['base_path'], burgers_config['data_name']), 'r')

    heat_train_data, heat_val_data, heat_test_data = get_data(heat_f, heat_config)

    KdV_train_data, KdV_val_data, KdV_test_data = get_data(KdV_f, KdV_config)

    burgers_train_data, burgers_val_data, burgers_test_data = get_data(burgers_f, burgers_config)

    fusion_loader = DataLoader(ConcatDataset([heat_val_data, KdV_val_data, burgers_val_data]), batch_size=128,
                               num_workers=0, generator=torch.Generator(device='cuda'), shuffle=False)

    heat_loader = DataLoader(heat_val_data, batch_size=128, num_workers=0, generator=torch.Generator(device='cuda'),
                             shuffle=False)

    burgers_loader = DataLoader(burgers_val_data, batch_size=128, num_workers=0,
                                generator=torch.Generator(device='cuda'), shuffle=False)

    KdV_loader = DataLoader(KdV_val_data, batch_size=128, num_workers=0, generator=torch.Generator(device='cuda'),
                            shuffle=False)

    loss_fn = nn.L1Loss(reduction='mean')

    # 1.四个模型在所有数据集上的表现，预测结论：必然是fusion最好
    # output：一组对照实验，四个模型在每一个任务上的loss对比、二十四张图片，四个模型在三个任务上的表现，每组两张
    # ==========================================
    # sub task 1:四个模型三个任务两个实例共二十四张图片
    fig, ax = plt.subplots(ncols=8, nrows=3, figsize=(5 * 8, 21))
    plt.text(3, 2, 'start')
    suptitle = ""
    for i, (dataloader, datatype) in enumerate([
        (heat_loader, "heat"),
        (burgers_loader, "burgers"),
        (KdV_loader, "KdV")
    ]):
        for j, (model, model_name) in enumerate([
            (fusion_model, "fusion"),
            (heat_model, "heat"),
            (burgers_model, "burgers"),
            (kdv_model, "kdv")
        ]):
            for x0, y, grid, tokens, t in dataloader:
                y_pred = model(grid.to(device), tokens.to(device), x0.to(device), t.to(device))
                y = y[..., 0].to(device=device)
                loss = round(loss_fn(y_pred, y).item(), 3)
                print(f"{datatype}_{model_name}: {loss}")
                ax[i][2 * j].plot(y[0].reshape(100, ).detach().cpu())
                ax[i][2 * j].plot(y_pred[0].reshape(100, ).detach().cpu())
                ax[i][2 * j].set_title(f"{datatype}_{model_name}_1")
                ax[i][2 * j + 1].plot(y[1].reshape(100, ).detach().cpu())
                ax[i][2 * j + 1].plot(y_pred[1].reshape(100, ).detach().cpu())
                ax[i][2 * j + 1].set_title(f"{datatype}_{model_name}_2")
                break

    plt.suptitle(suptitle, fontsize=36)
    plt.savefig("compare_models_1D.png")
    plt.close()

    # ==========================================
    # sub task 2:四个模型三个任务共十二个total_loss

    # ==========================================

    # 2.fusion和另外三个模型在各自数据集上的表现，预测结论：可能是fusion更好
    # output：三组对照实验，每一组对照试验的总loss、八个instance的图片结果
