import  os,sys,time, glob
import  numpy as np
import  torch
import  utils
import  logging
import  argparse
import  torch.nn as nn
from    torch import optim
import  torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from  model_search import Network
from datasets_csv import Imagefolder_csv


parser = argparse.ArgumentParser("miniImagenet")
parser.add_argument('--data', type=str, default='', help='/Datasets/miniImageNet/')
parser.add_argument('--lr', type=float, default=0.001, help='init learning rate')
parser.add_argument('--lr_min', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=5e-3, help='weight decay')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')
parser.add_argument('--exp_path', type=str, default='', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping range')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training/val splitting')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_lr', type=float, default=5e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_wd', type=float, default=1e-3, help='weight decay for arch encoding')
#  Few-shot parameters  #
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--episodeSize', type=int, default=1, help='the mini-batch size of training')
parser.add_argument('--episode_train_num', type=int, default=5000, help='the total number of training episodes')
parser.add_argument('--episode_test_num', type=int, default=1000, help='the total number of testing episodes')
parser.add_argument('--way_num', type=int, default=5, help='the number of way/class')
parser.add_argument('--shot_num', type=int, default=1, help='the number of shot')
parser.add_argument('--query_num', type=int, default=15, help='the number of queries')
parser.add_argument('--resume', default='', type=str, help='path to the lastest checkpoint (default: none)')
args = parser.parse_args()

utils.create_exp_dir(args.exp_path, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.exp_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


def adjust_learning_rate(optimizer, epoch_num):
    """Sets the learning rate to the initial LR decayed by 0.05 every 10 epochs"""
    lr = args.lr * (0.05 ** (epoch_num // 10))
    print("learning rate =", lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    np.random.seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)


    # ================================================
    total, used = os.popen(
        'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
            ).read().split('\n')[args.gpu].split(',')
    total = int(total)
    used = int(used)

    print('Total GPU mem:', total, 'used:', used)


    args.unrolled = False
    logging.info('GPU device = %d' % args.gpu)
    logging.info("args = %s", args)


    criterion = nn.CrossEntropyLoss().cuda()
    query_num = args.way_num * args.query_num
    model = Network(query_num, args.way_num, criterion).cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print("load successfully.")
    logging.info("Total param size = %f MB", utils.count_parameters_in_MB(model))

    # this is the optimizer to optimize
    # optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.9))


    ImgTransform = transforms.Compose([
        transforms.Resize((args.imageSize, args.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = Imagefolder_csv(
        data_dir=args.data, mode='train', image_size=args.imageSize, transform=ImgTransform,
        episode_num=args.episode_train_num, way_num=args.way_num, shot_num=args.shot_num, query_num=args.query_num
    )

    testset = Imagefolder_csv(
        data_dir=args.data, mode='test', image_size=args.imageSize, transform=ImgTransform,
        episode_num=args.episode_test_num, way_num=args.way_num, shot_num=args.shot_num, query_num=args.query_num
    )


    num_train = len(trainset)  #
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))  #

    num_test = len(testset)
    indices1 = list(range(num_test))


    train_queue = torch.utils.data.DataLoader(
        trainset, batch_size=args.episodeSize,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        trainset, batch_size=args.episodeSize,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=True, num_workers=0)

    test_queue = torch.utils.data.DataLoader(
        testset, batch_size=args.episodeSize,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices1[0:num_test]),
        pin_memory=True, num_workers=0)
    print("train_queue = ", len(train_queue))
    print("valid_queue = ", len(valid_queue))
    print("test_queue = ", len(test_queue))


    for epoch in range(args.epochs):


        lr = adjust_learning_rate(optimizer, epoch)
        logging.info('\nEpoch: %d lr: %e', epoch, lr)

        genotype = model.genotype()
        logging.info('Genotype: %s', genotype)

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, criterion, lr, optimizer, epoch)
        logging.info('train acc: %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(test_queue, model, criterion)
        logging.info('valid acc: %f', valid_acc)
        # scheduler.step()

        utils.save(model, os.path.join(args.exp_path, 'search'+str(epoch)+'.pt'))


def train(train_queue, valid_queue, model, criterion, lr, optimizer, epoch):
    """

    :param train_queue: train loader
    :param valid_queue: validate loader
    :param model: network
    :param arch: Arch class
    :param criterion:
    :param optimizer:
    :param lr:
    :return:
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    valid_iter = iter(valid_queue)
    optimizer1 = optim.Adam(model.arch_parameters(),
                            lr=args.arch_lr,
                            betas=(0.5, 0.999),
                            weight_decay=args.arch_wd)

    for step, (query_images, query_targets, support_images, support_targets) in enumerate(train_queue):


        model.train()

        query_images = torch.cat(query_images, 0)
        input_var1 = query_images.cuda()

        input_var2 = []
        for i in range(len(support_images)):
            temp_support = support_images[i]
            temp_support = torch.cat(temp_support, 0)
            temp_support = temp_support.cuda()
            input_var2.append(temp_support)

        # Deal with the targets
        target = torch.cat(query_targets, 0)
        target = target.cuda(non_blocking=True)


        query_images_search, query_targets_search, support_images_search, support_targets_search = next(valid_iter)
        query_images_search = torch.cat(query_images_search, 0)
        input_var1_search = query_images_search.cuda()

        input_var2_search = []
        for i in range(len(support_images_search)):
            temp_support_search = support_images_search[i]
            temp_support_search = torch.cat(temp_support_search, 0)
            temp_support_search = temp_support_search.cuda()
            input_var2_search.append(temp_support_search)

        # Deal with the targets
        target_search = torch.cat(query_targets_search, 0)
        target_search = target_search.cuda(non_blocking=True)
        batchsz = target.size(0)

        # 1. update alpha

        optimizer1.zero_grad()
        loss1 = model.loss(input_var1_search, input_var2_search, target_search)
        loss1.backward(retain_graph=True)
        optimizer1.step()


        logits = model(input_var1, input_var2)
        loss2 = criterion(logits, target)

        # 2. update weight
        optimizer.zero_grad()
        loss2.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        losses.update(loss2.item(), batchsz)
        top1.update(prec1.item(), batchsz)
        top5.update(prec5.item(), batchsz)

        if step % args.report_freq == 0:
            logging.info('epoch:%f Step:%03d loss:%f acc1:%f acc5:%f',epoch, step, losses.avg, top1.avg, top5.avg)
            genotype = model.genotype()
            logging.info('Genotype: %s', genotype)

    return top1.avg, losses.avg


def infer(valid_queue, model, criterion):
    """

    :param valid_queue:
    :param model:
    :param criterion:
    :return:
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (query_images, query_targets, support_images, support_targets) in enumerate(valid_queue):

            query_images = torch.cat(query_images, 0)
            input_var1 = query_images.cuda()

            input_var2 = []
            for i in range(len(support_images)):
                temp_support = support_images[i]
                temp_support = torch.cat(temp_support, 0)
                temp_support = temp_support.cuda()
                input_var2.append(temp_support)

            # Deal with the targets
            target = torch.cat(query_targets, 0)
            target = target.cuda(non_blocking=True)
            batchsz = target.size(0)

            logits = model(input_var1, input_var2)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), batchsz)
            top1.update(prec1.item(), batchsz)
            top5.update(prec5.item(), batchsz)

            if step % args.report_freq == 0:
                logging.info('>> Validation: %3d %e %f %f', step, losses.avg, top1.avg, top5.avg)

    return top1.avg, losses.avg


if __name__ == '__main__':
    main()
