import os
import numpy as np
import random
import argparse
import time
import json
import pandas as pd
from collections import Counter
from geomloss import SamplesLoss
from torch import Tensor
import math
from scipy.stats import t  # 导入scipy的t分布

import clip

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io

import sys
sys.path.append('.')

from src.modules import *
from src import logger


parser = argparse.ArgumentParser(description='FairCLIP Training/Fine-Tuning')

parser.add_argument('--seed', default=-1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float,
                    metavar='W', help='weight decay (default: 6e-5)',
                    dest='weight_decay')

parser.add_argument('--result_dir', default='./results', type=str)
parser.add_argument('--dataset_dir', default='./data', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--eval_set', default='test', type=str, help='options: val | test')
parser.add_argument('--summarized_note_file', default='', type=str)
parser.add_argument('--text_source', default='note', type=str, help='options: note | label')
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--model_arch', default='vit-b16', type=str, help='options: vit-b16 | vit-l14')
parser.add_argument('--pretrained_weights', default='', type=str)
parser.add_argument('--attributes', default='race,gender,ethnicity,language', type=str, 
                    help='comma-separated list of attributes to optimize for fairness: race|gender|ethnicity|language')
parser.add_argument('--batchsize_fairloss', default=64, type=int)
parser.add_argument('--lambda_fairloss', default=1e-4, type=float)
parser.add_argument('--sinkhorn_blur', default=1e-4, type=float)

# 添加滑动窗口窗口相关参数
parser.add_argument('--window_size', default=77, type=int, help='滑动窗口大小')
parser.add_argument('--window_overlap', default=20, type=int, help='窗口重叠大小')
parser.add_argument('--attention_lr', default=1e-4, type=float, help='注意力池化层学习率')

# 添加Student-t分布的自由度参数
parser.add_argument('--t_dist_dof', default=5, type=int, help='Student-t分布的自由度，通常取3-5')

def weight_estimation(loss, mean, std, dof):
    """
    使用Student-t分布计算权重
    参数:
        loss: 当前损失值
        mean: 损失的均值
        std: 损失的标准差
        dof: 自由度
    返回:
        基于Student-t分布的权重
    """
    # 标准化损失值
    t_stat = (loss - mean) / std
    # 计算Student-t分布的概率密度
    return t.pdf(t_stat, dof)

if __name__ == '__main__':
    args = parser.parse_args()
    args.seed=9129
    logger.log(f'===> random seed: {args.seed}')
    logger.log(f'===> Using Student-t distribution with degrees of freedom: {args.t_dist_dof}')

    logger.configure(dir=args.result_dir, log_suffix='train')

    with open(os.path.join(args.result_dir, f'args_train.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # 定义属性组数量和映射
    groups_in_attrs = [3, 2, 2, 3]  # race, gender, ethnicity, language
    attr_to_idx = {'race': 0, 'gender': 1, 'ethnicity': 2, 'language': 3}
    model_arch_mapping = {'vit-b16': 'ViT-B/16', 'vit-l14': 'ViT-L/14'}
    
    # 解析需要优化的属性列表
    args.attributes = args.attributes.split(',')
    target_attr_indices = [attr_to_idx[attr] for attr in args.attributes]
    logger.log(f'Optimizing fairness for attributes: {args.attributes} (indices: {target_attr_indices})')

    best_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'best_{args.perf_file}')
    acc_head_str = ''
    auc_head_str = ''
    dpd_head_str = ''
    eod_head_str = ''
    esacc_head_str = ''
    esauc_head_str = ''
    group_disparity_head_str = ''
    if args.perf_file != '':
        if not os.path.exists(best_global_perf_file):
            for i in range(len(groups_in_attrs)):
                auc_head_str += ', '.join([f'auc_attr{i}_group{x}' for x in range(groups_in_attrs[i])]) + ', '
            dpd_head_str += ', '.join([f'dpd_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            eod_head_str += ', '.join([f'eod_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            esacc_head_str += ', '.join([f'esacc_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            esauc_head_str += ', '.join([f'esauc_attr{x}' for x in range(len(groups_in_attrs))]) + ', '

            group_disparity_head_str += ', '.join([f'std_group_disparity_attr{x}, max_group_disparity_attr{x}' for x in range(len(groups_in_attrs))]) + ', '
            
            with open(best_global_perf_file, 'w') as f:
                f.write(f'epoch, acc, {esacc_head_str} auc, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_head_str} path\n')

    # 初始化设备和模型
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_arch_mapping[args.model_arch], device=device, jit=False)
    
    # 获取隐藏层维度
    hidden_dim = 512 if args.model_arch == 'vit-b16' else 768
    
    # 初始化注意力池化层，确保与模型数据类型一致
    attention_pool = TextEncoderWithAttentionPooling(model, hidden_dim=hidden_dim).to(device)

    # 初始化数据集 - 使用滑动窗口版本
    train_dataset = fair_vl_med_dataset(
        args.dataset_dir, 
        preprocess, 
        subset='Training', 
        text_source=args.text_source, 
        summarized_note_file=args.summarized_note_file,
        window_size=args.window_size,
        overlap=args.window_overlap
    )
    # 使用自定义collate_fn处理变长窗口
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=False,
        collate_fn=train_dataset.collate_fn
    )

    val_dataset = fair_vl_med_dataset(
        args.dataset_dir, 
        preprocess, 
        subset='Validation',
        window_size=args.window_size,
        overlap=args.window_overlap
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=False,
        collate_fn=val_dataset.collate_fn
    )

    test_dataset = fair_vl_med_dataset(
        args.dataset_dir, 
        preprocess, 
        subset='Test',
        window_size=args.window_size,
        overlap=args.window_overlap
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=False,
        collate_fn=test_dataset.collate_fn
    )
    
    logger.log(f'# of training samples: {train_dataset.__len__()}, # of testing samples: {test_dataset.__len__()}')
    
    # 初始化分组数据加载器 - 为每个属性创建单独的分组加载器
    attr_group_dataloaders = {}  # 结构: {属性索引: [组0加载器, 组1加载器, ...]}
    for attr_idx in target_attr_indices:
        attr_name = [k for k, v in attr_to_idx.items() if v == attr_idx][0]
        group_loaders = []
        for group_id in range(groups_in_attrs[attr_idx]):
            tmp_dataset = fair_vl_group_dataset(
                args.dataset_dir, 
                preprocess, 
                text_source='note', 
                summarized_note_file=args.summarized_note_file, 
                attribute=attr_name, 
                thegroup=group_id,
                window_size=args.window_size,
                overlap=args.window_overlap
            )
            tmp_dataloader = DataLoader(
                tmp_dataset, 
                batch_size=args.batchsize_fairloss, 
                shuffle=True,
                num_workers=args.workers, 
                pin_memory=True, 
                drop_last=False,
                collate_fn=tmp_dataset.collate_fn
            )
            group_loaders.append(endless_loader(tmp_dataloader))
        attr_group_dataloaders[attr_idx] = group_loaders
        logger.log(f'Created {len(group_loaders)} group loaders for attribute: {attr_name}')

    # 统计分组大小
    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(train_dataset)
    logger.log(f'group size on race in training set: {group_size_on_race}')
    logger.log(f'group size on gender in training set: {group_size_on_gender}')
    logger.log(f'group size on ethnicity in training set: {group_size_on_ethnicity}')
    group_size_on_race, group_size_on_gender, group_size_on_ethnicity = count_number_of_groups(test_dataset)
    logger.log(f'group size on race in test set: {group_size_on_race}')
    logger.log(f'group size on gender in test set: {group_size_on_gender}')
    logger.log(f'group size on ethnicity in test set: {group_size_on_ethnicity}')

    # 模型精度转换函数
    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 

    # 保持模型为Float类型以避免CLIP内部LayerNorm冲突
    model = model.float()
    attention_pool = attention_pool.float()

    # 损失函数和优化器
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    loss_for_FairCLIP = SamplesLoss(loss="sinkhorn", p=2, blur=args.sinkhorn_blur)
    
    # 优化器 - 为注意力池化层设置不同的学习率，避免参数重复
    # 先获取所有模型参数并标记，确保每个参数只属于一个组
    model_params = list(model.named_parameters())
    attention_params = list(attention_pool.named_parameters())

    # 创建参数组字典，使用参数ID作为键以避免重复
    param_groups = {}

    # 添加模型参数
    for name, param in model_params:
        param_groups[id(param)] = {"params": param, "lr": args.lr}

    # 添加注意力池化层参数（如果它们不在模型参数中）
    for name, param in attention_params:
        param_id = id(param)
        if param_id not in param_groups:
            param_groups[param_id] = {"params": param, "lr": args.attention_lr}

    # 转换为优化器需要的列表格式
    optimizer = optim.Adam(
        list(param_groups.values()),
        betas=(0.1, 0.1), 
        eps=1e-6, 
        weight_decay=args.weight_decay
    )

    # 加载预训练权重（如果有）
    if args.pretrained_weights != "":
        checkpoint = torch.load(args.pretrained_weights)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        attention_pool.load_state_dict(checkpoint['attention_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 确保加载后的数据类型为Float
        model = model.float()
        attention_pool = attention_pool.float()
    else:
        start_epoch = 0

    # 初始化最佳指标跟踪变量
    best_epoch = 0
    best_auc = sys.float_info.min
    best_acc = sys.float_info.min
    best_es_acc = sys.float_info.min
    best_es_auc = sys.float_info.min
    best_auc_groups = None
    best_dpd_groups = None
    best_eod_groups = None
    best_between_group_disparity = None
    
    # 初始化内存模块和历史训练记录
    num_batches = math.ceil(7000/args.batch_size)
    memory_module = np.zeros((num_batches, args.num_epochs))
    historical_training = np.zeros((num_batches, args.num_epochs))
    mean_epoch = 0
    std_epoch = 0
    weight_module = np.zeros((num_batches, args.num_epochs))
    
    threshold_noisy = 0
    threshold_faulty = 0
    noisy_param = 3
    faulty_param = 3
    mask = np.zeros((num_batches, 1), dtype=bool)

    # 训练循环
    for epoch in range(start_epoch, args.num_epochs):
        avg_loss = 0
        
        # 计算阈值和掩码（用于噪声检测）
        if (epoch > 0):
            historical_training[:, epoch] = (1/(epoch)) * np.sum(memory_module, axis=1)
            mean_epoch = np.mean(historical_training[:, epoch], axis = 0)
            std_epoch = np.std(historical_training[:, epoch], axis = 0)
            threshold_faulty = mean_epoch + faulty_param * std_epoch
            threshold_noisy = mean_epoch - noisy_param * std_epoch
            mask = (historical_training[:, epoch] < threshold_noisy) | (historical_training[:, epoch] > threshold_faulty)
            logger.log(f"Epoch {epoch} thresholds - faulty: {threshold_faulty:.4f}, noisy: {threshold_noisy:.4f}")
                
        counter = 0
        model.train()
        attention_pool.train()
                
        for i, batch in enumerate(train_dataloader) :
            optimizer.zero_grad()

            images, text_windows, label_and_attributes = batch 
            # 图像保持为Float类型
            images = images.to(device, dtype=torch.float32)
            # 关键修改：文本窗口需要是整数类型（Long），因为它们是token索引
            text_windows = text_windows.to(device, dtype=torch.long)

            # 提取图像特征
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=1)  # 归一化
            
            # 使用注意力池化聚合文本窗口特征
            text_features, attn_weights = attention_pool(text_windows, image_features)
            
            # 计算对比损失
            logits = (image_features @ text_features.T) * torch.exp(torch.tensor(0.07, device=device, dtype=torch.float32))
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            total_loss = (loss_img(logits, ground_truth) + loss_txt(logits.T, ground_truth)) / 2
            
            # 应用噪声/错误样本权重 - 使用Student-t分布
            if (mask[i] and epoch > 0):
                # 传入Student-t分布的自由度参数
                weight = weight_estimation(total_loss.item(), mean_epoch, std_epoch, args.t_dist_dof)
                total_loss = total_loss * weight
            
            memory_module[i, epoch] = total_loss.item()
            
            # 计算公平性损失 - 对所有目标属性计算并累加
            similarity = (image_features @ text_features.T)
            correlations_with_batch = similarity.diag().float()
            
            # 为每个属性计算公平性损失
            for attr_idx in target_attr_indices:
                attr_name = [k for k, v in attr_to_idx.items() if v == attr_idx][0]
                group_loaders = attr_group_dataloaders[attr_idx]
                
                # 对该属性的每个分组计算损失
                for group_loader in group_loaders:
                    images_dist, texts_dist, label_and_attributes_dist = next(group_loader)
                    images_dist = images_dist.to(device, dtype=torch.float32)
                    texts_dist = texts_dist.to(device, dtype=torch.long)
                    
                    with torch.no_grad():
                        # 计算组分布的图像特征
                        img_feats_dist = model.encode_image(images_dist)
                        img_feats_dist = F.normalize(img_feats_dist, dim=1)
                        
                        # 计算组分布的文本特征（带注意力池化）
                        txt_feats_dist, _ = attention_pool(texts_dist, img_feats_dist)
                    
                    # 计算相似度并归一化
                    similarity_dist = (img_feats_dist @ txt_feats_dist.T)
                    correlations_with_group = similarity_dist.diag().float()
                    correlations_with_group /= correlations_with_group.sum()
                    
                    # 累加公平性损失，对每个属性使用相同的权重
                    total_loss += (args.lambda_fairloss / len(target_attr_indices)) * loss_for_FairCLIP(
                        correlations_with_batch[:,None], 
                        correlations_with_group[:,None]
                    )
            
            # 反向传播和优化
            total_loss.backward()
            optimizer.step()
            
            avg_loss += total_loss.item()
            counter += 1
            
        avg_loss /= len(train_dataloader)
        logger.log(f"# of batches processed: {counter}")
        
        # 评估阶段
        model.eval()
        attention_pool.eval()
        eval_avg_loss = 0
        all_probs = []
        all_labels = []
        all_attrs = []
        
        with torch.no_grad():
            for batch in test_dataloader :
                images, text_windows, label_and_attributes = batch 

                images = images.to(device, dtype=torch.float32)
                # 评估阶段同样确保文本窗口为Long类型
                text_windows = text_windows.to(device, dtype=torch.long)
                glaucoma_labels = label_and_attributes[:, 0].to(device)
                attributes = label_and_attributes[:, 1:].to(device)

                # 提取图像特征
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=1)

                # 处理两类文本（非青光眼/青光眼）
                class_text_feats = []
                for i in range(text_windows.shape[1]):  # text_windows形状: (batch_size, 2, num_windows, seq_len)
                    # 对每个类别应用注意力池化
                    txt_feats, _ = attention_pool(text_windows[:, i, :, :], image_features)
                    class_text_feats.append(txt_feats[:, None, :])
                
                # 合并类别特征
                class_text_feats = torch.cat(class_text_feats, dim=1)
                
                # 计算概率
                vl_prob, vl_logits = compute_vl_prob(image_features, class_text_feats)
                
                all_probs.append(vl_prob[:,1].cpu().numpy())
                all_labels.append(glaucoma_labels.cpu().numpy())
                all_attrs.append(attributes.cpu().numpy())

                # 计算评估损失
                loss = F.binary_cross_entropy(vl_prob[:,1].float(), glaucoma_labels.float())
                eval_avg_loss += loss.item()

        # 处理评估结果
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_attrs = np.concatenate(all_attrs, axis=0)
        eval_avg_loss /= len(test_dataloader)

        logger.log(f'===> epoch[{epoch:03d}/{args.num_epochs:03d}], training loss: {avg_loss:.4f}, eval loss: {eval_avg_loss:.4f}')

        # 评估综合性能
        overall_acc, eval_es_acc, overall_auc, eval_es_auc, eval_aucs_by_attrs, eval_dpds, eval_eods, between_group_disparity = evalute_comprehensive_perf(all_probs, all_labels, all_attrs.T)

        # 更新最佳模型
        if best_auc <= overall_auc:
            best_auc = overall_auc
            best_acc = overall_acc
            best_ep = epoch
            best_auc_groups = eval_aucs_by_attrs
            best_dpd_groups = eval_dpds
            best_eod_groups = eval_eods
            best_es_acc = eval_es_acc
            best_es_auc = eval_es_auc
            best_between_group_disparity = between_group_disparity
        
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'attention_state_dict': attention_pool.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': eval_avg_loss,
                }, os.path.join(args.result_dir, f"clip_ep{epoch:03d}.pth"))

        # 保存预测结果
        if args.result_dir is not None:
            np.savez(os.path.join(args.result_dir, f'pred_gt_ep{epoch:03d}.npz'), 
                        val_pred=all_probs, val_gt=all_labels, val_attr=all_attrs)

        # 记录日志
        logger.log(f'---- best AUC {best_auc:.4f} at epoch {best_ep}')
        logger.log(f'---- best AUC by groups and attributes at epoch {best_ep}')
        logger.log(best_auc_groups)

        logger.logkv('epoch', epoch)
        logger.logkv('trn_loss', round(avg_loss,4))
        logger.logkv('eval_loss', round(eval_avg_loss,4))
        logger.logkv('eval_acc', round(overall_acc,4))
        logger.logkv('eval_auc', round(overall_auc,4))

        for ii in range(len(eval_es_acc)):
            logger.logkv(f'eval_es_acc_attr{ii}', round(eval_es_acc[ii],4))
        for ii in range(len(eval_es_auc)):
            logger.logkv(f'eval_es_auc_attr{ii}', round(eval_es_auc[ii],4))
        for ii in range(len(eval_aucs_by_attrs)):
            for iii in range(len(eval_aucs_by_attrs[ii])):
                logger.logkv(f'eval_auc_attr{ii}_group{iii}', round(eval_aucs_by_attrs[ii][iii],4))

        for ii in range(len(between_group_disparity)):
            logger.logkv(f'eval_auc_attr{ii}_std_group_disparity', round(between_group_disparity[ii][0],4))
            logger.logkv(f'eval_auc_attr{ii}_max_group_disparity', round(between_group_disparity[ii][1],4))

        for ii in range(len(eval_dpds)):
            logger.logkv(f'eval_dpd_attr{ii}', round(eval_dpds[ii],4))
        for ii in range(len(eval_eods)):
            logger.logkv(f'eval_eod_attr{ii}', round(eval_eods[ii],4))

        logger.dumpkvs()
            
    # 保存最佳性能结果
    if args.perf_file != '':
        if os.path.exists(best_global_perf_file):
            with open(best_global_perf_file, 'a') as f:
                esacc_head_str = ', '.join([f'{x:.4f}' for x in best_es_acc]) + ', '
                esauc_head_str = ', '.join([f'{x:.4f}' for x in best_es_auc]) + ', '

                auc_head_str = ''
                for i in range(len(best_auc_groups)):
                    auc_head_str += ', '.join([f'{x:.4f}' for x in best_auc_groups[i]]) + ', '

                group_disparity_str = ''
                for i in range(len(best_between_group_disparity)):
                    group_disparity_str += ', '.join([f'{x:.4f}' for x in best_between_group_disparity[i]]) + ', '
                
                dpd_head_str = ', '.join([f'{x:.4f}' for x in best_dpd_groups]) + ', '
                eod_head_str = ', '.join([f'{x:.4f}' for x in best_eod_groups]) + ', '

                path_str = f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}'
                f.write(f'{best_ep}, {best_acc:.4f}, {esacc_head_str} {best_auc:.4f}, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_str} {path_str}\n')

    os.rename(args.result_dir, f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}')
