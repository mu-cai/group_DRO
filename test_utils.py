import argparse
import torchvision as tv
from torchvision import transforms
import torch
import numpy as np
import sklearn.metrics as sk
from models import model_attributes


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_datadir", help="Path to the in-distribution data folder.")
    parser.add_argument("--out_datadir", help="Path to the out-of-distribution data folder.")

    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")

    parser.add_argument("--logdir", required=True,
                        help="Where to log test info (small).")
    parser.add_argument("--batch", type=int, default=256,
                        help="Batch size.")
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring and checkpointing.")

    parser.add_argument("--model", default="BiT-S-R101x1", help="Which variant to use")
    parser.add_argument("--model_path", type=str, help="Path to the finetuned model you want to test")
    parser.add_argument('--feature_ouput', action='store_true', help='OE define OOD data as positive.')
    parser.add_argument('--tiny_name',  type=str, default='cifar_10_train_tiny_dict.pt', help='Folder to save checkpoints.')
    parser.add_argument('--save_dis_name',  type=str, default='distance_dist', help='Folder to save checkpoints.')
    parser.add_argument('--single_k', action='store_true', help='Test only flag.')
    parser.add_argument('--p',  type=int, default=2, help='norm')
    parser.add_argument('--k',  type=int, default=1, help='Folder to save checkpoints.')
    parser.add_argument('--specififc_k', action='store_true', help='Test only flag.')
    parser.add_argument('--use_cuda', action='store_true', help='Test only flag.')
    parser.add_argument('--batch_size', '-b', type=int, default=1000, help='Batch size.')
    parser.add_argument('--regression', action='store_true', help='Test only flag.')
    parser.add_argument('--regress_folder', default='snapshots_dist/', type=str, help='regress path')
    parser.add_argument('--regress_path', default='_wrn_s1_distance_epoch_9_linear.pt', type=str, help='regress path')
    parser.add_argument('--dict_size', type=int, default=50000, help='Batch size.')  # 50000 = 50k
    parser.add_argument('--save_score', action='store_true', help='Test only flag.')
    parser.add_argument('--ood_folder', default='snapshots_dist/', type=str, help='regress path')
    parser.add_argument('--use_val_id', action='store_true', help='Test only flag.')


    return parser


def mk_id_ood(args, logger):
    """Returns train and validation datasets."""
    crop = 480

    # val_tx = tv.transforms.Compose([
    #     tv.transforms.Resize((crop, crop)),
    #     tv.transforms.ToTensor(),
    #     tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])

    val_tx = transforms.Compose([
        transforms.Resize(model_attributes[args.model]['target_resolution']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # in_set = tv.datasets.ImageFolder(args.in_datadir, val_tx)
    out_set = tv.datasets.ImageFolder(args.out_datadir, val_tx)

    # logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    # logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

    # in_loader = torch.utils.data.DataLoader(
    #     in_set, batch_size=args.batch, shuffle=False,
    #     num_workers=args.workers, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.batch_size, pin_memory=True, drop_last=False)
        

    # return in_set, out_set, in_loader, out_loader
    return out_loader


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=1.):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(in_examples, out_examples):
    num_in = in_examples.shape[0]
    num_out = out_examples.shape[0]

    # logger.info("# in example is: {}".format(num_in))
    # logger.info("# out example is: {}".format(num_out))

    labels = np.zeros(num_in + num_out, dtype=np.int32)
    labels[:num_in] += 1

    examples = np.squeeze(np.vstack((in_examples, out_examples)))
    aupr_in = sk.average_precision_score(labels, examples)
    auroc = sk.roc_auc_score(labels, examples)

    recall_level = 0.95
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    labels_rev = np.zeros(num_in + num_out, dtype=np.int32)
    labels_rev[num_in:] += 1
    examples = np.squeeze(-np.vstack((in_examples, out_examples)))
    aupr_out = sk.average_precision_score(labels_rev, examples)
    return auroc, aupr_in, aupr_out, fpr
