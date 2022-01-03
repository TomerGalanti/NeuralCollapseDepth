import gc
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models

from tqdm import tqdm
from collections import OrderedDict
from scipy.sparse.linalg import svds
from torchvision import datasets, transforms
from IPython import embed
import torch


def analysis(graphs, model, settings, loader, train=False):
    model.eval()

    initialized_results = False
    num_embs = 0
    N = []
    mean = []
    mean_s = []

    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(loader, start=1):
            if data.shape[0] != settings.batch_size:
                continue

            data, target = data.to(settings.device), target.to(settings.device)
            _, embeddings = model(data)
            num_embs = len(embeddings)

            if initialized_results == False: # if we did not init means to zero
                N = [settings.num_output_classes*[0] for _ in range(num_embs)]
                mean = [settings.num_output_classes*[0] for _ in range(num_embs)]
                mean_s = [settings.num_output_classes*[0] for _ in range(num_embs)]
                initialized_results = True

            for i in range(num_embs): # which top layer to investigate
                h = embeddings[i]

                for c in range(settings.num_output_classes):
                    idxs = (target == c).nonzero(as_tuple=True)[0]
                    if len(idxs) == 0:  # If no class-c in this batch
                        continue

                    h_c = h[idxs, :]
                    mean[i][c] += torch.sum(h_c, dim=0)
                    N[i][c] += h_c.shape[0]
                    mean_s[i][c] += torch.sum(torch.square(h_c))


        for i in range(num_embs):
            for c in range(settings.num_output_classes):
                mean[i][c] /= N[i][c]
                mean_s[i][c] /= N[i][c]

            avg_cdnv = 0
            total_num_pairs = settings.num_output_classes * (settings.num_output_classes - 1) / 2
            for class1 in range(settings.num_output_classes):
                for class2 in range(class1 + 1, settings.num_output_classes):
                    variance1 = abs(mean_s[i][class1].item() - torch.sum(torch.square(mean[i][class1])).item())
                    variance2 = abs(mean_s[i][class2].item() - torch.sum(torch.square(mean[i][class2])).item())
                    variance_avg = (variance1 + variance2) / 2
                    dist = torch.norm((mean[i][class1]) - (mean[i][class2]))**2
                    dist = dist.item()
                    cdnv = variance_avg / dist
                    avg_cdnv += cdnv / total_num_pairs

            if train:
                graphs.cdnv_train[i].append(avg_cdnv)
            else:
                graphs.cdnv_test[i].append(avg_cdnv)


def embedding_performance(graphs, model, settings, train_loader, test_loader):

    model.eval()
    data = next(iter(train_loader))[0].to(settings.device)
    embeddings = model(data)[1]
    num_embs = len(embeddings)
    linear_projs = []
    loss_function = nn.CrossEntropyLoss()
    params = list()

    for i in range(num_embs):

        emb = embeddings[i]
        emb = emb.view(emb.size()[0], -1)
        emb_dim = emb.shape[1]

        # init the linear classifiers
        linear_proj = nn.Linear(emb_dim, settings.num_output_classes).to(settings.device)
        linear_projs += [linear_proj]
        params += list(linear_proj.parameters())

    # init the optimizer
    optimizer = optim.SGD(params, lr=settings.top_lr, momentum=0.9, weight_decay=5e-4)

    # train phase
    for batch_idx, (data, target) in enumerate(train_loader, start=1):

        if data.shape[0] != settings.batch_size:
            continue

        loss = 0.0
        data, target = data.to(settings.device), target.to(settings.device)
        _, embeddings = model(data)

        optimizer.zero_grad()

        for i in range(num_embs):
            outputs = linear_projs[i](embeddings[i])
            loss += loss_function(outputs, target)

        loss.backward()
        optimizer.step()

    # test phase
    test_losses = num_embs*[0.0]
    corrects = num_embs*[0.0]

    for (images, labels) in test_loader:

        if settings.device == 'cuda':
            images = images.cuda()
            labels = labels.cuda()

        for i in range(num_embs):
            _, embeddings = model(images)
            outputs = linear_projs[i](embeddings[i])
            test_losses[i] += loss_function(outputs, labels).item()
            _, preds = outputs.max(1)
            corrects[i] += preds.eq(labels).sum().item()

    dataset_size = len(test_loader.dataset)
    accuracy_rates = [corrects[i] / dataset_size for i in range(num_embs)]
    test_losses = [test_losses[i] / dataset_size for i in range(num_embs)]

    for i in range(num_embs):
        graphs.emb_accuracy[i].append(accuracy_rates[i])
        graphs.emb_loss[i].append(test_losses[i])