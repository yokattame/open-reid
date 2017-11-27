from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint


def merge(name, train, val, trainval, numbers, data_dir, split_id):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    shift = numbers[0]
    for element in dataset.train:
        fname, pid, cam = element
        train.append((name + '/images/' + fname, pid + shift, cam))
    numbers[0] += dataset.num_train_ids

    if name == 'cuhk03' or name == 'market1501':
        shift = numbers[1]
        for element in dataset.val:
            fname, pid, cam = element
            val.append((name + '/images/' + fname, pid + shift, cam))
        numbers[1] += dataset.num_val_ids
    
    shift = numbers[2]
    for element in dataset.trainval:
        fname, pid, cam = element
        trainval.append((name + '/images/' + fname, pid + shift, cam))
    numbers[2] += dataset.num_trainval_ids

    return dataset


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)
    
    train, val, trainval = [], [], []
    numbers = [0, 0, 0]

    dataset_cuhk03 = merge('cuhk03', train, val, trainval, numbers, args.data_dir, args.split)
    dataset_market1501 = merge('market1501', train, val, trainval, numbers, args.data_dir, args.split)
    merge('cuhksysu', train, val, trainval, numbers, args.data_dir, args.split)
    merge('mars', train, val, trainval, numbers, args.data_dir, args.split)
    
    num_train_ids, num_val_ids, num_trainval_ids = numbers
    
    assert num_val_ids == dataset_cuhk03.num_val_ids + dataset_market1501.num_val_ids

    print("============================================")
    print("JSTL dataset loaded")
    print("  subset   | # ids | # images")
    print("  ---------------------------")
    print("  train    | {:5d} | {:8d}"
          .format(num_train_ids, len(train)))
    print("  val      | {:5d} | {:8d}"
          .format(num_val_ids, len(val)))
    print("  trainval | {:5d} | {:8d}"
          .format(num_trainval_ids, len(trainval)))

    query_cuhk03, gallery_cuhk03 = dataset_cuhk03.query, dataset_cuhk03.gallery
    query_market1501, gallery_market1501 = dataset_market1501.query, dataset_market1501.gallery

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = trainval if args.combine_trainval else train
    num_classes = (num_trainval_ids if args.combine_trainval
                   else num_train_ids)

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(args.height, args.width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_set, root=args.data_dir,
                     transform=train_transformer),
        batch_size=args.batch_size, num_workers=args.workers,
        sampler=RandomIdentitySampler(train_set, args.num_instances),
        pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(val, root=args.data_dir,
                     transform=test_transformer),
        batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, pin_memory=True)

    test_loader_cuhk03 = DataLoader(
        Preprocessor(list(set(query_cuhk03) | set(gallery_cuhk03)),
                     root=dataset_cuhk03.images_dir, transform=test_transformer),
        batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, pin_memory=True)

    test_loader_market1501 = DataLoader(
        Preprocessor(list(set(query_market1501) | set(gallery_market1501)),
                     root=dataset_market1501.images_dir, transform=test_transformer),
        batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, pin_memory=True)

    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)
    model = models.create(args.arch, num_features=1024,
                          dropout=args.dropout, num_classes=args.features)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    model = nn.DataParallel(model).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        metric.train(model, train_loader)
        print("Validation:")
        evaluator.evaluate(val_loader, val, val, metric)
        print("Test(cuhk03):")
        evaluator.evaluate(test_loader_cuhk03, query_cuhk03, gallery_cuhk03, metric)
        print("Test(market1501):")
        evaluator.evaluate(test_loader_market1501, query_market1501, gallery_market1501, metric)
        return

    # Criterion
    criterion = TripletLoss(margin=args.margin).cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Trainer
    trainer = Trainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr if epoch <= 100 else \
            args.lr * (0.001 ** ((epoch - 100) / 50.0))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)
        if epoch < args.start_save:
            continue
        top1 = evaluator.evaluate(val_loader, val, val)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    metric.train(model, train_loader)

    print("Test(cuhk03):")
    evaluator.evaluate(test_loader_cuhk03, query_cuhk03, gallery_cuhk03, metric)
    print("Test(market1501):")
    evaluator.evaluate(test_loader_market1501, query_market1501, gallery_market1501, metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="JSTL Triplet loss classification")
    # data
    #parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
    #                    choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())

