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

    # logger.info("Processing in-distribution data...")
    in_scores = iterate_data_msp(in_loader, model)
    # logger.info("Processing out-of-distribution data...")
    out_scores = iterate_data_msp(out_loader, model)


    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    
    print('AUROC: {}'.format(auroc), '\n', aupr_in,'\n',  aupr_out, '\n', fpr95 )

    # logger.info('============Results for {}============'.format(args.score))
    # logger.info('AUROC: {}'.format(auroc))
    # logger.info('AUPR (In): {}'.format(aupr_in))
    # logger.info('AUPR (Out): {}'.format(aupr_out))
    # logger.info('FPR95: {}'.format(fpr95))

    # logger.flush()





def iterate_data_msp(data_loader, model):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


