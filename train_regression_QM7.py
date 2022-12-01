"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import importlib
import shutil
import argparse

from pathlib import Path
# from tqdm import tqdm
from data_utils.MoleculeDataSet import PointCloudMoleculeDataSet, load_and_align_QM7

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))




def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    # parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    # parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--data_fp', help='Where to find the QM7 dataset file')
    parser.add_argument('--results_fp', help='Where to store a txt file of results')
    parser.add_argument('--n_train', type=int)
    parser.add_argument('--n_test', type=int)
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def write_result_to_file(fp: str, missing_str: str='', **trial) -> None:
    """Write a line to a tab-separated file saving the results of a single
        trial.
    Parameters
    ----------
    fp : str
        Output filepath
    missing_str : str
        (Optional) What to print in the case of a missing trial value
    **trial : dict
        One trial result. Keys will become the file header
    Returns
    -------
    None
    """
    header_lst = list(trial.keys())
    header_lst.sort()
    if not os.path.isfile(fp):
        header_line = "\t".join(header_lst) + "\n"
        with open(fp, 'w') as f:
            f.write(header_line)
    trial_lst = [str(trial.get(i, missing_str)) for i in header_lst]
    trial_line = "\t".join(trial_lst) + "\n"
    with open(fp, 'a') as f:
        f.write(trial_line)


def test(model: torch.nn.Module, 
            loader: torch.utils.data.DataLoader, 
            loss: torch.nn.Module,
            device: torch.device) -> torch.Tensor:


    # device = model.device
    # device = torch.device('cpu')
    model_eval = model.eval()

    all_preds = []
    all_targets = []
    for (points_and_features, U_matrices, target) in loader:

        points_and_features = points_and_features.to(device)
        U_matrices = U_matrices.to(device)
        # target = target.to(device)
        preds = model_eval(points_and_features, U_matrices)

        all_preds.append(preds.cpu())
        all_targets.append(target)

    all_preds = torch.cat(all_preds).flatten()
    all_targets = torch.cat(all_targets).flatten()

    return loss(all_preds, all_targets)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('regression_QM7')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, 'pointnet2_reg_msg'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    train_dset, val_dset, test_dset = load_and_align_QM7(fp=args.data_fp,
                                                            n_train=args.n_train,
                                                            n_test=args.n_test,
                                                            validation_set_fraction=0.1)
    n_train = len(train_dset)
    n_val = len(val_dset)
    n_test = len(test_dset)

    # train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    # test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size, shuffle=True)

    # trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    model = importlib.import_module('pointnet2_reg_msg')

    '''Copy the effective files into the log directory. Im not sure how I feel about this...'''
    shutil.copy('./data_utils/MoleculeDataSet.py', str(exp_dir))
    shutil.copy('./models/pointnet2_reg_msg.py', str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_regression_QM7.py', str(exp_dir))
    results_logging_fp = os.path.join(exp_dir, 'epoch_results.txt')

    '''DECIDE WHICH DEVICE TO TRAIN ON'''
    bool_use_CUDA = torch.cuda.is_available()

    # The following line is copied from https://stackoverflow.com/questions/50954479/using-cuda-with-pytorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    '''COMPILE MODEL AND LOSS FUNCTION'''
    classifier = model.get_model(n_centroids_1=10,
                                    msg_radii_1=[2., 4., 8.],
                                    msg_nsample_1=[4, 8, 16],
                                    n_centroids_2=4,
                                    msg_radii_2=[2., 4., 8.],
                                    msg_nsample_2=[2, 4, 8],
                                    in_channels=5,
                                    out_channels=128,
                                    device=device)
    classifier.apply(inplace_relu)
    
    criterion = model.get_loss()
    

    '''MOVE EVERYTHING TO THE CORRECT DEVICE'''

    classifier = classifier.to(device)
    criterion = criterion.to(device)



    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except FileNotFoundError:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_val_MSE = torch.Tensor(np.inf)
    # best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        # mean_correct = []
        classifier = classifier.train()

        training_losses = []
        # for batch_id, (points, features, target) in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
        for batch_id, (points_and_features, U_matrices, target) in enumerate(train_loader):

            # print(f"Points has shape {points.shape}")
            # print(f"Features has shape {features.shape}")
            # print(f"Target has shape {target.shape}")

            optimizer.zero_grad()

            # points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            # points = torch.Tensor(points)

            if bool_use_CUDA:
                points_and_features = points_and_features.cuda()
                U_matrices = U_matrices.cuda()
                target = target.cuda()

            pred = classifier(points_and_features, U_matrices)
            loss = criterion(pred, target)
            training_losses.append(loss.cpu().data)
            loss.backward()
            optimizer.step()
            global_step += 1

        train_MSE_for_epoch = np.mean(training_losses)

        # train_instance_acc = np.mean(mean_correct)
        # log_string('Train MSE: %f' % train_MSE_for_epoch)
        scheduler.step()

        with torch.no_grad():
            val_MSE = test(classifier.eval(), val_loader, criterion, device)
            log_string('Train MSE: %f. Val MSE: %f' % (train_MSE_for_epoch, val_MSE))
            # instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            test_MSE = test(classifier.eval(), test_loader, criterion, device)



            if (val_MSE <= best_val_MSE):
                log_string('New best val MSE')
                best_val_MSE = val_MSE
                best_epoch = epoch + 1

                log_string('Test MSE: %f' % test_MSE)

            # log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            # log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            # if (instance_acc >= best_instance_acc):
                logger.info('Saving model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                # log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'train_MSE': train_MSE_for_epoch,
                    'test_MSE': test_MSE,
                    'val_MSE': val_MSE,
                    'train_idxes': train_dset.idxes,
                    'val_idxes': val_dset.idxes,
                    'test_idxes': test_dset.idxes,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

            logging_dd = {
                'train_MSE': train_MSE_for_epoch.item(),
                'test_MSE': test_MSE.item(),
                'val_MSE': val_MSE.item(),
                'epoch': epoch,
                'n_train': n_train,
                'n_val': n_val,
                'n_test': n_test
            }

            write_result_to_file(results_logging_fp, **logging_dd)
    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
