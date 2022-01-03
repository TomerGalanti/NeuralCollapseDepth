# train.py
#!/usr/bin/env	python3

""" train network using pytorch
author baiyu
"""

import os
import time
from shutil import copyfile
import operator

import torch
import torch.nn as nn
import torch.optim as optim
import analysis


from conf import global_settings as settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

class graphs:
    def __init__(self):
        self.train_accuracy = []
        self.test_accuracy = []

        self.train_loss = []
        self.test_loss = []

        # NC1
        self.cdnv_train = []
        self.cdnv_test = []

        # Emb performance
        self.emb_accuracy = []
        self.emb_loss = []
        self.num_embs = 0


def train(epoch, train_loader):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(train_loader):

        if settings.device == 'cuda':
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs, _ = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * settings.batch_size + len(images),
            total_samples=len(train_loader.dataset)
        ))

        #update training loss for each iteration
        #writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= settings.warm:
            warmup_scheduler.step()

    # for name, param in net.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))



@torch.no_grad()
def eval_training(epoch, test_loader):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:

        if settings.device == 'cuda':
            images = images.cuda()
            labels = labels.cuda()

        outputs, _ = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()

    dataset_size = len(test_loader.dataset)
    acc = correct / dataset_size
    test_loss = test_loss / dataset_size

    finish = time.time()
    if settings.device == 'cuda':
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss,
        acc,
        finish - start
    ))
    print()

    #add informations to tensorboard
    # if tb:
    #     writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    #     writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return acc, test_loss

if __name__ == '__main__':


    net = get_network(settings)

    ## results arrays
    graphs = graphs()
    attrbts = [attr for attr in dir(graphs) if not \
        callable(getattr(graphs, attr)) and not attr.startswith("__")]

    ## save data configs
    if not os.path.isdir(settings.directory):
        os.mkdir(settings.directory)
        os.mkdir(settings.directory + '/' + '0')

    sub_dirs_ids = [x[0] for x in os.walk(settings.directory)][1:]
    sub_dirs_ids = [int(dir[len(settings.directory) + 1:]) for dir in sub_dirs_ids]

    xid = max(sub_dirs_ids) + 1
    dir_name = settings.directory + '/' + str(xid)
    os.mkdir(dir_name)

    ## save hyperparams
    copyfile('conf/global_settings.py', dir_name + '/settings.py')

    #data preprocessing:
    train_loader = get_training_dataloader(
        settings.dataset_name,
        settings.mean,
        settings.std,
        num_workers=2,
        batch_size=settings.batch_size,
        shuffle=True
    )

    test_loader = get_test_dataloader(
        settings.dataset_name,
        settings.mean,
        settings.std,
        num_workers=2,
        batch_size=settings.batch_size,
        shuffle=True
    )

    # compute the number of embedding layers
    _, embeddings = net(next(iter(train_loader))[0].to(settings.device))
    graphs.num_embs = num_embs = len(embeddings)

    graphs.cdnv_train = [[] for _ in range(num_embs)]
    graphs.cdnv_test = [[] for _ in range(num_embs)]
    graphs.emb_accuracy = [[] for _ in range(num_embs)]
    graphs.emb_loss = [[] for _ in range(num_embs)]

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=settings.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * settings.warm)

    if settings.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, settings.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, settings.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, settings.net, settings.TIME_NOW)

    #use tensorboard
    # if not os.path.exists(settings.LOG_DIR):
    #     os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    # writer = SummaryWriter(log_dir=os.path.join(
    #         settings.LOG_DIR, args.net, settings.TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if settings.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, settings.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, settings.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, settings.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, settings.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, settings.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > settings.warm:
            train_scheduler.step(epoch)

        if settings.resume:
            if epoch <= resume_epoch:
                continue

        analysis.analysis(graphs, net, settings, train_loader, train=True)
        analysis.analysis(graphs, net, settings, test_loader, train=False)
        analysis.embedding_performance(graphs, net, settings, train_loader, test_loader)

        train(epoch, train_loader)

        train_acc, train_loss = eval_training(epoch, train_loader)
        test_acc, test_loss = eval_training(epoch, test_loader)


        ## keep track of the results
        graphs.train_loss += [train_loss]
        graphs.test_loss += [test_loss]
        graphs.train_accuracy += [train_acc]
        graphs.test_accuracy += [test_acc]

        ## save the results
        for name in attrbts:
            if name in ['cdnv_train', 'cdnv_test', 'emb_accuracy', 'emb_loss']:
                for i in range(num_embs):
                    _ = open(dir_name + '/' + name + '_' + str(i) + '.txt', 'w+')
                    _.write(str(operator.attrgetter(name)(graphs)[i]))
                    _.close()
            else:
                _ = open(dir_name + '/' + name + ".txt", "w+")
                _.write(str(operator.attrgetter(name)(graphs)))
                _.close()

        #start to save best performance model after learning rate decay to 0.01
        # if epoch > settings.MILESTONES[1] and best_acc < acc:
        #     weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
        #     print('saving weights file to {}'.format(weights_path))
        #     torch.save(net.state_dict(), weights_path)
        #     best_acc = acc
        #     continue
        #
        # if not epoch % settings.SAVE_EPOCH:
        #     weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
        #     print('saving weights file to {}'.format(weights_path))
        #     torch.save(net.state_dict(), weights_path)

    #writer.close()