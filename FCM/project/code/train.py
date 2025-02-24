import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from contrastive_losses import *
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from feature_memory import *
from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/FCM', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    labeled_bs = args.labeled_bs

    feature_memory = FeatureMemory(num_samples=args.labeled_bs, memory_per_class=256, feature_size=16,
                                    n_classes=args.num_classes)

    interp = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    stu_model1 = create_model()
    stu_model2 = create_model()
    tea_model = create_model()
    tea_model.detach_model()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    stu_model1.train()
    stu_model2.train()
    tea_model.train()
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.SGD(stu_model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(stu_model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    mse_criterion = nn.MSELoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    loss_contr_unlabeled1 = 0
    loss_contr_unlabeled2 = 0
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    cur_threshold = 1 / num_classes
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs1 = stu_model1(volume_batch)
            outputs2 = stu_model2(volume_batch)
            outputs3 = tea_model(volume_batch)

            outputs_soft1 = torch.softmax(outputs1[0], dim=1)
            outputs_soft2 = torch.softmax(outputs2[0], dim=1)
            outputs_soft3 = torch.softmax(outputs3[0], dim=1)


            # pseudo_output1 = torch.argmax(outputs_soft1, dim=1)
            # pseudo_output2 = torch.argmax(outputs_soft2, dim=1)

            pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:], dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:], dim=1, keepdim=False)
            pseudo_outputs3 = torch.argmax(outputs_soft3[args.labeled_bs:], dim=1, keepdim=False)

            with torch.no_grad():
                max_values1, _ = torch.max(outputs_soft1, dim=1)
                max_values2, _ = torch.max(outputs_soft2, dim=1)
                percent = (iter_num + 1) / max_iterations

                cur_threshold1 = (1 - percent) * cur_threshold + percent * max_values1.mean()
                cur_threshold2 = (1 - percent) * cur_threshold + percent * max_values2.mean()
                mean_max_values = min(max_values1.mean(), max_values2.mean())

                cur_threshold = min(cur_threshold1, cur_threshold2)
                cur_threshold = torch.clip(cur_threshold, 0.25, 0.95)

            mask_high = (outputs_soft1 > cur_threshold) & (outputs_soft2 > cur_threshold)
            mask_non_similarity = (mask_high == False)
            high_outputs_soft1 = torch.mul(mask_high, outputs_soft1)
            high_outputs_soft2 = torch.mul(mask_high, outputs_soft2)

            low_outputs_soft1 = torch.mul(mask_non_similarity, outputs_soft1)
            low_outputs_soft2 = torch.mul(mask_non_similarity, outputs_soft2)


            high_pseudo_outputs1 = torch.argmax(high_outputs_soft1, dim=1, keepdim=False)
            high_pseudo_outputs2 = torch.argmax(high_outputs_soft2, dim=1, keepdim=False)


            loss_dice1 = dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss1 = loss_dice1

            loss_dice2 = dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss2 = loss_dice2

            loss_mts1 = dice_loss(outputs_soft1[args.labeled_bs:], pseudo_outputs3.unsqueeze(1).detach())
            loss_mts2 = dice_loss(outputs_soft2[args.labeled_bs:], pseudo_outputs3.unsqueeze(1).detach())

            loss_mse = mse_criterion(low_outputs_soft1[args.labeled_bs:], low_outputs_soft2[args.labeled_bs:])

            if mean_max_values >= 0.95:
                loss_cps1 = dice_loss(
                    outputs_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1).detach())
                loss_cps2 = dice_loss(
                    outputs_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1).detach())

            else:
                loss_cps1 = dice_loss(
                    high_outputs_soft1[args.labeled_bs:], high_pseudo_outputs2[args.labeled_bs:].unsqueeze(1).detach())
                loss_cps2 = dice_loss(
                    high_outputs_soft2[args.labeled_bs:], high_pseudo_outputs1[args.labeled_bs:].unsqueeze(1).detach())



            loss_cps = 0.5 * (loss_cps1 + loss_cps2)

            loss_mts = 0.5 * (loss_mts1 + loss_mts2)

            consistency_weight = get_current_consistency_weight(iter_num // 150)


            if iter_num > 2000:
                with torch.no_grad():
                    labeled_pred = outputs3[0]
                    labeled_features = outputs3[1]
                    labeled_pred = labeled_pred[:args.labeled_bs]
                    labeled_features = labeled_features[:args.labeled_bs]

                    labeled_pred = interp(labeled_pred)
                    probability_prediction, label_prediction = torch.max(torch.softmax(labeled_pred, dim=1),
                                                                         dim=1)  # Get pseudolabels

                # Resize labels, predictions and probabilities,  to feature map resolution
                labels_down = nn.functional.interpolate(label_batch[:][:args.labeled_bs].float().unsqueeze(1),
                                                        size=(labeled_features.shape[2], labeled_features.shape[3]),
                                                        mode='nearest').squeeze(1)
                label_prediction_down = nn.functional.interpolate(label_prediction.float().unsqueeze(1), size=(
                labeled_features.shape[2], labeled_features.shape[3]), mode='nearest').squeeze(1)
                probability_prediction_down = nn.functional.interpolate(probability_prediction.float().unsqueeze(1),
                                                                        size=(labeled_features.shape[2],
                                                                              labeled_features.shape[3]),
                                                                        mode='nearest').squeeze(1)

                # get mask where the labeled predictions are correct and have a confidence higher than 0.95
                mask_prediction_correctly = ((label_prediction_down == labels_down).float() * (
                            probability_prediction_down > 0.95).float()).bool()
                # Apply the filter mask to the features and its labels

                labeled_features_correct = labeled_features.permute(0, 2, 3, 1)

                labels_down_correct = labels_down[mask_prediction_correctly]

                labeled_features_correct = labeled_features_correct[mask_prediction_correctly, ...]

                # get projected features
                with torch.no_grad():
                    proj_labeled_features_correct = tea_model.projection_head(labeled_features_correct)

                # updated memory bank
                feature_memory.add_features_from_sample_learned(tea_model, proj_labeled_features_correct,
                                                                 labels_down_correct, labeled_bs)



            if iter_num > 2000:
                '''
                CONTRASTIVE LEARNING ON UNLABELED DATA. align unlabeled features to labeled features
                '''
                unlabeled_pred = outputs1[0]
                unlabeled_pred = unlabeled_pred[args.labeled_bs:]
                unprobability_prediction, unlabel_prediction = torch.max(torch.softmax(unlabeled_pred, dim=1),
                                                                         dim=1)  # Get pseudolabels
                unlabeled_features = outputs1[1]

                unlabeled_features = unlabeled_features[args.labeled_bs:]
                pseudolabels_down = nn.functional.interpolate(pseudo_outputs1.float().unsqueeze(1),
                                                              size=(unlabeled_features.shape[2],
                                                                    unlabeled_features.shape[3]),
                                                              mode='nearest').squeeze(1)

                # mask features that do not have ignore label in the labels (zero-padding because of data augmentation like resize/crop)

                mask = ((probability_prediction_down > 0).float()).bool()

                unlabeled_features = unlabeled_features.permute(0, 2, 3, 1)
                unlabeled_features = unlabeled_features[mask, ...]
                pseudolabels_down = pseudolabels_down[mask]
                # get predicted features

                proj_feat_unlabeled = stu_model1.projection_head(unlabeled_features)
                pred_feat_unlabeled = stu_model1.prediction_head(proj_feat_unlabeled)

                # Apply contrastive learning loss
                loss_contr_unlabeled1 = contrastive_class_to_class_learned_memory(stu_model1, pred_feat_unlabeled,
                                                                                  pseudolabels_down,
                                                                                  num_classes, feature_memory.memory)

                unlabeled_pred = outputs2[0]
                unlabeled_pred = unlabeled_pred[args.labeled_bs:]
                unprobability_prediction, unlabel_prediction = torch.max(torch.softmax(unlabeled_pred, dim=1),
                                                                         dim=1)  # Get pseudolabels
                unlabeled_features = outputs2[1]

                unlabeled_features = unlabeled_features[args.labeled_bs:]
                pseudolabels_down = nn.functional.interpolate(pseudo_outputs2.float().unsqueeze(1),
                                                              size=(unlabeled_features.shape[2],
                                                                    unlabeled_features.shape[3]),
                                                              mode='nearest').squeeze(1)

                # mask features that do not have ignore label in the labels (zero-padding because of data augmentation like resize/crop)

                mask = ((probability_prediction_down > 0).float()).bool()

                unlabeled_features = unlabeled_features.permute(0, 2, 3, 1)
                unlabeled_features = unlabeled_features[mask, ...]
                pseudolabels_down = pseudolabels_down[mask]
                # get predicted features

                proj_feat_unlabeled = stu_model2.projection_head(unlabeled_features)
                pred_feat_unlabeled = stu_model2.prediction_head(proj_feat_unlabeled)

                # Apply contrastive learning loss
                loss_contr_unlabeled2 = contrastive_class_to_class_learned_memory(stu_model2, pred_feat_unlabeled,
                                                                                  pseudolabels_down,
                                                                                  num_classes, feature_memory.memory)
            loss = supervised_loss1 + supervised_loss2 + consistency_weight * (loss_mts + loss_cps) + consistency_weight * (500 * loss_mse) + 0.6 * loss_contr_unlabeled1 + 0.6 * loss_contr_unlabeled2
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            tea_model.weighted_update(stu_model1, stu_model2, ema_decay=0.99, cur_step=iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/supervised_loss1',
                              supervised_loss1, iter_num)
            writer.add_scalar('loss/supervised_loss2',
                              supervised_loss2, iter_num)
            writer.add_scalar('loss/contrast_loss1',
                              loss_contr_unlabeled1, iter_num)
            writer.add_scalar('loss/contrast_loss2',
                              loss_contr_unlabeled2, iter_num)
            logging.info(
                'iteration %d : supervised_loss1 : %f supervised_loss2 : %f  loss_contr_unlabeled1 : %f   loss_contr_unlabeled2 : %f  loss_cps : %f  loss_mse : %f' % (
                iter_num, supervised_loss1.item(), supervised_loss2.item(), loss_contr_unlabeled1, loss_contr_unlabeled2, loss_cps, loss_mse))

            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1[0], dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                stu_model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], stu_model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(stu_model1.state_dict(), save_mode_path)
                    torch.save(stu_model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                stu_model1.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(stu_model1.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + '/log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
