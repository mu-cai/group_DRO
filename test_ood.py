# from utils import log
import torch
import time
import os

import numpy as np

from test_utils import arg_parser, mk_id_ood, get_measures
import os

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable



def get_ood_value(model, in_loader, out_loader, logger=None, args=None, num_classes=None, train_loader_in=None):
    model.eval()

    if args.energy:
        in_scores = iterate_data_energy(in_loader, model)
        out_scores = iterate_data_energy(out_loader, model)
    else:
        # logger.info("Processing in-distribution data...")
        in_scores = iterate_data_msp(in_loader, model)
        # logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_msp(out_loader, model)


    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    auroc, aupr_in, aupr_out, fpr95 = auroc*100, aupr_in*100, aupr_out*100, fpr95*100
    
    print('AUROC: {:.3}'.format(auroc), '\n', 'AUPR (In): {:.3}'.format(aupr_in),'\n',  'AUPR (Out): {:.3}'.format(aupr_out), '\n', 'FPR95: {:.3}'.format(fpr95) )
    return auroc, aupr_in, aupr_out, fpr95 

    # logger.info('============Results for {}============'.format(args.score))
    # logger.info('AUROC: {}'.format(auroc))
    # logger.info('AUPR (In): {}'.format(aupr_in))
    # logger.info('AUPR (Out): {}'.format(aupr_out))
    # logger.info('FPR95: {}'.format(fpr95))

    # logger.flush()





# def iterate_data_msp(data_loader, model):
#     confs = []
#     m = torch.nn.Softmax(dim=-1).cuda()
#     for b, (x, y) in enumerate(data_loader):
#         with torch.no_grad():
#             x = x.cuda()
#             # compute output, measure accuracy and record loss.
#             logits = model(x)

#             conf, _ = torch.max(m(logits), dim=-1)
#             confs.extend(conf.data.cpu().numpy())
#     return np.array(confs)


def iterate_data_msp(data_loader, model):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    # for b, (x, y) in enumerate(data_loader):
    for batch_idx, batch in enumerate(data_loader):
        with torch.no_grad():
            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            # y = batch[1]
            # g = batch[2]
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_energy(data_loader, model):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    # for b, (x, y) in enumerate(data_loader):
    for batch_idx, batch in enumerate(data_loader):
        with torch.no_grad():
            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            # y = batch[1]
            # g = batch[2]
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf =  torch.logsumexp(logits, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)



