import h5py
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torch import nn
from transformers import BertConfig, BertModel

from models import pitt_cls, pitt_cls_lhs
from models.pitt import PhysicsInformedTokenTransformer
from models.pitt_bert import PhysicsInformedTokenTransformerBertFullHs0
from train_pitt_1d_fusion import get_neural_operator

from torch.utils.data import DataLoader, ConcatDataset

from utils import TransformerOperatorDataset, TransformerOperatorDatasetBert

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')


def get_data(f, config):
    # train_data = TransformerOperatorDataset(f, config['flnm'],
    #                                         split="train",
    #                                         initial_step=config['initial_step'],
    #                                         reduced_resolution=config['reduced_resolution'],
    #                                         reduced_resolution_t=config['reduced_resolution_t'],
    #                                         reduced_batch=config['reduced_batch'],
    #                                         saved_folder=config['base_path'],
    #                                         return_text=config['return_text'],
    #                                         num_t=config['num_t'],
    #                                         num_x=config['num_x'],
    #                                         sim_time=config['sim_time'],
    #                                         num_samples=config['num_samples'],
    #                                         train_style=config['train_style'],
    #                                         rollout_length=config['rollout_length'],
    #                                         seed=config['seed'],
    #                                         val_ratio=0.2,
    #                                         )
    # train_data.data = train_data.data.to(device)
    # train_data.grid = train_data.grid.to(device)
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
                                          val_ratio=1,
                                          test_ratio=0,
                                          )
    val_data.data = val_data.data.to(device)
    val_data.grid = val_data.grid.to(device)
    # test_data = TransformerOperatorDataset(f, config['flnm'],
    #                                        split="test",
    #                                        initial_step=config['initial_step'],
    #                                        reduced_resolution=config['reduced_resolution'],
    #                                        reduced_resolution_t=config['reduced_resolution_t'],
    #                                        reduced_batch=config['reduced_batch'],
    #                                        saved_folder=config['base_path'],
    #                                        return_text=config['return_text'],
    #                                        num_t=config['num_t'],
    #                                        num_x=config['num_x'],
    #                                        sim_time=config['sim_time'],
    #                                        num_samples=config['num_samples'],
    #                                        train_style=config['train_style'],
    #                                        rollout_length=config['rollout_length'],
    #                                        seed=config['seed'],
    #                                        val_ratio=0.2,
    #                                        )
    # test_data.data = test_data.data.to(device)
    # test_data.grid = test_data.grid.to(device)
    # return train_data, val_data, test_data
    return val_data


def get_data_bert(f, config):
    # train_data = TransformerOperatorDatasetBert(f, config['flnm'],
    #                                             split="train",
    #                                             initial_step=config['initial_step'],
    #                                             reduced_resolution=config['reduced_resolution'],
    #                                             reduced_resolution_t=config['reduced_resolution_t'],
    #                                             reduced_batch=config['reduced_batch'],
    #                                             saved_folder=config['base_path'],
    #                                             return_text=config['return_text'],
    #                                             num_t=config['num_t'],
    #                                             num_x=config['num_x'],
    #                                             sim_time=config['sim_time'],
    #                                             num_samples=config['num_samples'],
    #                                             train_style=config['train_style'],
    #                                             rollout_length=config['rollout_length'],
    #                                             seed=config['seed'],
    #                                             val_ratio=0.2,
    #                                             )
    # train_data.data = train_data.data.to(device)
    # train_data.grid = train_data.grid.to(device)
    val_data = TransformerOperatorDatasetBert(f, config['flnm'],
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
                                              val_ratio=1,
                                              test_ratio=0
                                              )
    val_data.data = val_data.data.to(device)
    val_data.grid = val_data.grid.to(device)
    # test_data = TransformerOperatorDatasetBert(f, config['flnm'],
    #                                            split="test",
    #                                            initial_step=config['initial_step'],
    #                                            reduced_resolution=config['reduced_resolution'],
    #                                            reduced_resolution_t=config['reduced_resolution_t'],
    #                                            reduced_batch=config['reduced_batch'],
    #                                            saved_folder=config['base_path'],
    #                                            return_text=config['return_text'],
    #                                            num_t=config['num_t'],
    #                                            num_x=config['num_x'],
    #                                            sim_time=config['sim_time'],
    #                                            num_samples=config['num_samples'],
    #                                            train_style=config['train_style'],
    #                                            rollout_length=config['rollout_length'],
    #                                            seed=config['seed'],
    #                                            val_ratio=0.2,
    #                                            )
    # test_data.data = test_data.data.to(device)
    # test_data.grid = test_data.grid.to(device)
    # return train_data, val_data, test_data
    return val_data


def get_model(model_path, config):
    neural_operator = get_neural_operator(config['neural_operator'], config)
    model = PhysicsInformedTokenTransformer(500, config['hidden'], config['layers'], config['heads'],
                                            config['num_x'], dropout=config['dropout'],
                                            neural_operator=neural_operator).to(device=device)
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict['model_state_dict'])
    return model


def get_model_bert_hs0(model_path, config):
    model_config = BertConfig.from_pretrained('models/BERT/bert-tiny/bert_config.json')
    bert_model_path = 'models/BERT/bert-tiny'
    # 修改配置
    model_config.output_hidden_states = True
    model_config.output_attentions = False
    # 通过配置和路径导入模型
    bert_model = BertModel.from_pretrained(bert_model_path, config=model_config, ignore_mismatched_sizes=True)
    neural_operator = get_neural_operator(config['neural_operator'], config)
    model = PhysicsInformedTokenTransformerBertFullHs0(500, config['hidden'], config['layers'], config['heads'],
                                                       config['num_x'], dropout=config['dropout'],
                                                       neural_operator=neural_operator, bert_model=bert_model).to(
        device=device)
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict['model_state_dict'])
    return model


def get_model_bert_cls(model_path, config):
    model_config = BertConfig.from_pretrained('models/BERT/bert-tiny/bert_config.json')
    bert_model_path = 'models/BERT/bert-tiny'
    # 修改配置
    model_config.output_hidden_states = True
    model_config.output_attentions = False
    # 通过配置和路径导入模型
    bert_model = BertModel.from_pretrained(bert_model_path, config=model_config, ignore_mismatched_sizes=True)
    neural_operator = get_neural_operator(config['neural_operator'], config)
    model = pitt_cls.PhysicsInformedTokenTransformerBertCls(500, config['hidden'], config['layers'], config['heads'],
                                                            config['num_x'], dropout=config['dropout'],
                                                            neural_operator=neural_operator, bert_model=bert_model).to(
        device=device)
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict['model_state_dict'])
    return model


def get_model_bert_cls_lhs(model_path, config):
    model_config = BertConfig.from_pretrained('models/BERT/bert-tiny/bert_config.json')
    bert_model_path = 'models/BERT/bert-tiny'
    # 修改配置
    model_config.output_hidden_states = True
    model_config.output_attentions = False
    # 通过配置和路径导入模型
    bert_model = BertModel.from_pretrained(bert_model_path, config=model_config, ignore_mismatched_sizes=True)
    neural_operator = get_neural_operator(config['neural_operator'], config)
    model = pitt_cls_lhs.PhysicsInformedTokenTransformerBertCls(500, config['hidden'], config['layers'],
                                                                config['heads'],
                                                                config['num_x'], dropout=config['dropout'],
                                                                neural_operator=neural_operator,
                                                                bert_model=bert_model).to(
        device=device)
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict['model_state_dict'])
    return model


if __name__ == '__main__':

    # fusion_10_model_path = '1D_results/pitt_oformer_Fusion_varied_next_step_novel_32_0.001_0.0001_0_10/Fusion_pitt_32_1e-4_1e-6_0.0.pt'
    # fusion_100_model_path = '1D_results/pitt_oformer_Fusion_varied_next_step_novel_32_0.001_0.0001_0_100/Fusion_pitt_32_1e-4_1e-6_0.0.pt'
    fusion_1000_model_path = '1D_results/pitt_oformer_Burgers_varied_next_step_novel_bert_1000/Fusion_pitt_32_1e-4_1e-6_0.0.pt'
    #
    # heat_10_model_path = '1D_results/pitt_oformer_Heat_varied_next_step_novel_10/Heat_pitt_0.pt'
    # heat_100_model_path = '1D_results/pitt_oformer_Heat_varied_next_step_novel_100/Heat_pitt_0.pt'
    # heat_1000_model_path = '1D_results/pitt_oformer_Heat_varied_next_step_novel_1000/Heat_pitt_0.pt'
    #
    # burgers_10_model_path = '1D_results/pitt_oformer_Burgers_varied_next_step_novel_10/Burgers_pitt_0.pt'
    # burgers_100_model_path = '1D_results/pitt_oformer_Burgers_varied_next_step_novel_100/Burgers_pitt_0.pt'
    # burgers_1000_model_path = '1D_results/pitt_oformer_Burgers_varied_next_step_novel_1000/Burgers_pitt_0.pt'
    # bert_1000_model_lhs_path = '1D_results/pitt_oformer_Burgers_varied_next_step_novel_bert_1000/FusionBert_pitt_32_1e-4_1e-6_0.0_lhs_frozen.pt'
    # bert_1000_cls_lhs_model_path = '1D_results/pitt_oformer_Burgers_varied_next_step_novel_bert_1000/FusionBert_pitt_32_1e-4_1e-6_0.0_lhs_cls_frozen.pt'
    # bert_1000_cls_lhs_unfrozen_model_path = '1D_results/pitt_oformer_Burgers_varied_next_step_novel_bert_1000/FusionBert_pitt_32_1e-4_1e-6_0.0_lhs_unfrozen.pt'
    bert_1000_full_hs0_unfrozen_fine_tuned_model_path = '1D_results/pitt_oformer_Burgers_varied_next_step_novel_bert_1000/FusionBert_pitt_32_1e-4_1e-6_0.0_hs0_unfrozen_300.pt'
    # bert_1000_cls_lhs_unfrozen_fine_tuned_model_path = '1D_results/pitt_oformer_Burgers_varied_next_step_novel_bert_1000/FusionBert_pitt_32_1e-4_1e-6_0.0_lhs_cls_frozen_fine_tuning.pt'

    # kdv_10_model_path = '1D_results/pitt_oformer_KdV_varied_next_step_novel_10/KdV_pitt_0.pt'
    # kdv_100_model_path = '1D_results/pitt_oformer_KdV_varied_next_step_novel_100/KdV_pitt_0.pt'
    # kdv_1000_model_path = '1D_results/pitt_oformer_KdV_varied_next_step_novel_1000/KdV_pitt_0.pt'

    # heat_model_path = '1D_results/pitt_oformer_Heat_varied_next_step_novel/Heat_pitt_0.pt'
    # burgers_model_path = '1D_results/pitt_oformer_Burgers_varied_next_step_novel/Burgers_pitt_0.pt'
    # kdv_model_path = '1D_results/pitt_oformer_KdV_varied_next_step_novel/KdV_pitt_0.pt'

    with open("./configs/pitt_config_fusion.yaml", 'r') as stream:
        train_args = yaml.safe_load(stream)
    config = train_args['args']
    bert_config = config.copy()
    bert_config['hidden'] = 128
    num_samples = 1000

    # load models 因为用到的config的参数值没什么区别 所以此处用了同一个config
    # fusion_10_model = get_model(fusion_10_model_path, config)
    #
    # fusion_100_model = get_model(fusion_100_model_path, config)
    #
    fusion_1000_model = get_model(fusion_1000_model_path, config)
    #
    # heat_10_model = get_model(heat_10_model_path, config)
    #
    # heat_100_model = get_model(heat_100_model_path, config)
    #
    # heat_1000_model = get_model(heat_1000_model_path, config)
    #
    # burgers_10_model = get_model(burgers_10_model_path, config)
    #
    # burgers_100_model = get_model(burgers_100_model_path, config)
    #
    # burgers_1000_model = get_model(burgers_1000_model_path, config)

    # bert_1000_lhs_model = get_model_bert_cls_lhs(bert_1000_model_lhs_path, bert_config)
    #
    # bert_1000_cls_lhs_model = get_model_bert_cls_lhs(bert_1000_cls_lhs_model_path, bert_config)
    #
    # bert_1000_cls_lhs_unfrozen_model = get_model_bert_cls_lhs(bert_1000_cls_lhs_unfrozen_model_path, bert_config)
    #
    # bert_1000_cls_lhs_frozen_fine_tuned_model = get_model_bert_cls_lhs(bert_1000_cls_lhs_unfrozen_fine_tuned_model_path,
    #                                                                    bert_config)
    #
    bert_1000_full_hs0_frozen_fine_tuned_model = get_model_bert_hs0(bert_1000_full_hs0_unfrozen_fine_tuned_model_path,
                                                                    bert_config)
    #
    # kdv_10_model = get_model(kdv_10_model_path, config)
    #
    # kdv_100_model = get_model(kdv_100_model_path, config)
    #
    # kdv_1000_model = get_model(kdv_1000_model_path, config)

    # config for data loading process
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
        'num_samples': num_samples,
        'train_style': 'next_step',
        'rollout_length': 1,
        'seed': 0,

        'batch_size': 32,
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
        'num_samples': num_samples,
        'train_style': 'next_step',
        'rollout_length': 1,
        'seed': 0,

        'batch_size': 32,
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
        'num_samples': num_samples,
        'train_style': 'next_step',
        'rollout_length': 1,
        'seed': 0,

        'batch_size': 32,
        'num_workers': 0,

    }

    # data files
    heat_f = h5py.File("{}{}".format(heat_config['base_path'], heat_config['data_name']), 'r')
    KdV_f = h5py.File("{}{}".format(KdV_config['base_path'], KdV_config['data_name']), 'r')
    burgers_f = h5py.File("{}{}".format(burgers_config['base_path'], burgers_config['data_name']), 'r')

    # get data
    heat_val_data = get_data(heat_f, heat_config)

    KdV_val_data = get_data(KdV_f, KdV_config)

    burgers_val_data = get_data(burgers_f, burgers_config)

    heat_val_bert_data = get_data_bert(heat_f, heat_config)

    KdV_val_bert_data = get_data_bert(KdV_f, KdV_config)

    burgers_val_bert_data = get_data_bert(burgers_f, burgers_config)

    # dataloader construction
    # fusion_loader = DataLoader(ConcatDataset([heat_val_data, KdV_val_data, burgers_val_data]), batch_size=128,
    #                            num_workers=0, generator=torch.Generator(device='cuda'), shuffle=False)

    heat_loader = DataLoader(heat_val_data, batch_size=32, num_workers=0, generator=torch.Generator(device='cuda'),
                             shuffle=False)

    burgers_loader = DataLoader(burgers_val_data, batch_size=32, num_workers=0,
                                generator=torch.Generator(device='cuda'), shuffle=False)

    KdV_loader = DataLoader(KdV_val_data, batch_size=32, num_workers=0, generator=torch.Generator(device='cuda'),
                            shuffle=False)

    heat_bert_loader = DataLoader(heat_val_bert_data, batch_size=32, num_workers=0,
                                  generator=torch.Generator(device='cuda'),
                                  shuffle=False)

    burgers_bert_loader = DataLoader(burgers_val_bert_data, batch_size=32, num_workers=0,
                                     generator=torch.Generator(device='cuda'), shuffle=False)

    KdV_bert_loader = DataLoader(KdV_val_bert_data, batch_size=32, num_workers=0,
                                 generator=torch.Generator(device='cuda'),
                                 shuffle=False)

    # MAE loss
    loss_fn = nn.L1Loss(reduction='mean')

    # Output to file
    file = open("./compare_result.txt", "w+")

    # 1.四个模型在所有数据集上的表现，预测结论：必然是fusion最好
    # output：一组对照实验，四个模型在每一个任务上的loss对比、二十四张图片，四个模型在三个任务上的表现，每组两张
    # ==========================================
    # sub task 1:四个模型三个任务两个实例共二十四张图片
    # fig, ax = plt.subplots(ncols=8, nrows=3, figsize=(5 * 8, 21))
    # plt.text(3, 2, 'start')
    # suptitle = ""
    for j, (model, model_name) in enumerate([
        # (fusion_10_model, "fusion 10"),
        # (fusion_100_model, "fusion 100"),
        (fusion_1000_model, "fusion 1000"),
        # (heat_10_model, "heat 10"),
        # (heat_100_model, "heat 100"),
        # (heat_1000_model, "heat 1000"),
        # (burgers_10_model, "burgers 10"),
        # (burgers_100_model, "burgers 100"),
        # (burgers_1000_model, "burgers 1000"),
        # (bert_1000_lhs_model, "bert lhs 1000"),
        # (bert_1000_cls_lhs_model, "bert cls lhs 1000"),
        # (bert_1000_cls_lhs_unfrozen_model, "bert cls lhs unfrozen 1000"),
        # (bert_1000_cls_lhs_frozen_fine_tuned_model, "bert cls lhs frozen fine tuned 1000"),
        (bert_1000_full_hs0_frozen_fine_tuned_model, "bert full hs0 frozen fine tuned 1000"),
        # (kdv_10_model, "kdv 10"),
        # (kdv_100_model, "kdv 100"),
        # (kdv_1000_model, "kdv 1000"),
    ]):
        file.write(f"============evaluating Model:{model_name}============\n")
        if 'bert' in model_name:
            for n, (dataloader, datatype) in enumerate([
                (heat_bert_loader, "heat"),
                (burgers_bert_loader, "burgers"),
                (KdV_bert_loader, "KdV")
            ]):
                model.eval()
                with torch.no_grad():
                    loss = 0
                    data_num = int(0)
                    for x0, y, grid, tokens, t in dataloader:
                        y_pred = model(grid.to(device), tokens.to(device), x0.to(device), t.to(device), device=device)
                        y = y[..., 0].to(device=device)
                        data_num += x0.shape[0]
                        loss += loss_fn(y_pred, y).item() * x0.shape[0]
                    file.write(f"{datatype}: {loss / data_num}\n")
        else:
            for n, (dataloader, datatype) in enumerate([
                (heat_loader, "heat"),
                (burgers_loader, "burgers"),
                (KdV_loader, "KdV")
            ]):
                model.eval()
                with torch.no_grad():
                    loss = 0
                    for n, (x0, y, grid, tokens, t) in enumerate(dataloader):
                        y_pred = model(grid.to(device), tokens.to(device), x0.to(device), t.to(device))
                        y = y[..., 0].to(device=device)
                        loss += loss_fn(y_pred, y).item()
                    file.write(f"{datatype}: {loss / n}\n")
