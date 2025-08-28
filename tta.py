import argparse
import os
import math

# import ruamel.yaml as yaml
from ruamel.yaml import YAML
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from prettytable import PrettyTable

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_retrieval import APTM_Retrieval
from models.tokenization_bert import BertTokenizer

import utils
# from dataset import create_dataset, create_sampler, create_loader
from dataset.re_dataset import TextMaskingGenerator
from scheduler import create_scheduler
from optim import create_optimizer

from trains import train, train_attr
from train_pa100ks import train_pa100k, train_pa100k_only_img_classifier

from reTools import evaluation, mAP
from reTools import evaluation_attr, itm_eval_attr
from reTools import evaluation_attr_only_img_classifier, itm_eval_attr_only_img_classifier


from torch.cuda.amp import GradScaler


from tta.dataset import create_tta_dataset, create_tta_loader, create_dataset, create_loader
from tta.optim import configure_tta_model, create_tta_optimizer, create_tta_scheduler
from tta.adapt import test_time_adapt_itm
from tta.utils import preprocess_tta_coefficients
from tta.eval import evaluation_itc, evaluation_itm



# os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def main(args, config):
    utils.init_distributed_mode(args)

    print("### Hyper-parameters:")

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    print("     seed:", seed)

    device = torch.device(args.device)
    print("     device:", device)

    tbs = ["epoch", "R1", "R5", "R10", "mAP", "mINP"]
    table = PrettyTable(tbs)
    for tb in tbs[1:]:
        table.custom_format[tb] = lambda f, v: f"{v:.3f}"

    print("     output_dir:", args.output_dir)

    if args.bs > 0:
        config['batch_size_tta'] = args.bs
    if args.epo > 0:
        config['schedular']['epochs'] = args.epo
    if args.lr > 0:
        config['optimizer']['lr'] = args.lr
        config['schedular']['lr'] = args.lr
    print("     batch_size_tta:", config['batch_size_tta'])
    print("     epochs:", config['schedular']['epochs'])
    print("     lr:", config['schedular']['lr'])


    print("### Creating model")
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    model = APTM_Retrieval(config=config)
    if config['load_pretrained']:
        model.load_pretrained(args.checkpoint, config, is_eval=args.tta)
    model = model.to(device)
    print("     Total Params Sum: ", sum(p.numel() for p in model.parameters()))# if p.requires_grad))


    print("### Creating test dataset")
    if args.task == "itr_icfg":
        train_dataset, test_dataset = create_dataset('re_icfg', config, False, args.tta)
    elif args.task == "itr_rstp":
        train_dataset, val_dataset, test_dataset = create_dataset('re_rstp', config, False, args.tta)
    elif args.task == "itr_cuhk":
        train_dataset, val_dataset, test_dataset = create_dataset('re_cuhk', config, False, args.tta)
    # elif args.task == "itr_pa100k":
    #     train_dataset, val_dataset, test_dataset = create_dataset('re_pa100k', config, False, args.tta)
    # else:
    #     train_dataset, val_dataset, test_dataset = create_dataset('re_gene', config, False, args.tta)
    print(f"     test_dataset: {len(test_dataset)}")
    # sample = next(iter(test_dataset))
    # print(sample) #sample[0].shape=([3, 384, 128]) sample[1] = int(0)

    print("### Creating test dataloader")
    test_loader = create_loader(
        [test_dataset],
        batch_size=[config['batch_size_test']],
        num_workers=[4],
        is_trains=[False],
        collate_fns=[None]
    )[0]
    print(f"     test_loader: {len(test_loader)}")
    # batch = next(iter(test_loader))
    # print(batch)#batch[0].shape=([150, 3, 384, 128]), batch[1].shape=([150])


    print("### Inference ITC similiarity matrix")
    ## run only at first time to avoid error, then using np.load() to load itm input features.
    sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(
        model,
        test_loader,
        tokenizer,
        device,
        config,
    )
    if args.task == "itr_icfg":
        if "pretrain" in args.config:
            temp_feature_dir = "APTM_TTA_ICFG_pretrain"
        else:
            temp_feature_dir = "APTM_TTA_ICFG"
    elif args.task == "itr_rstp":
        if "pretrain" in args.config:
            temp_feature_dir = "APTM_TTA_RSTP_pretrain"
        else:
            temp_feature_dir = "APTM_TTA_RSTP"
    elif args.task == "itr_cuhk":
        if "pretrain" in args.config:
            temp_feature_dir = "APTM_TTA_CUHK_pretrain"
        else:
            temp_feature_dir = "APTM_TTA_CUHK"
    os.makedirs(f"/data/jiahao/{temp_feature_dir}/debug_embeddings/", exist_ok=True)
    np.save(f"/data/jiahao/{temp_feature_dir}/debug_embeddings/sims_matrix_t2i.npy", sims_matrix_t2i.detach().cpu().numpy())
    np.save(f"/data/jiahao/{temp_feature_dir}/debug_embeddings/image_embeds.npy", image_embeds.detach().cpu().numpy())
    np.save(f"/data/jiahao/{temp_feature_dir}/debug_embeddings/text_embeds.npy", text_embeds.detach().cpu().numpy())
    np.save(f"/data/jiahao/{temp_feature_dir}/debug_embeddings/text_atts.npy", text_atts.detach().cpu().numpy())
    sims_matrix_t2i = torch.from_numpy(np.load(f"/data/jiahao/{temp_feature_dir}/debug_embeddings/sims_matrix_t2i.npy"))#.to(device)
    image_embeds = torch.from_numpy(np.load(f"/data/jiahao/{temp_feature_dir}/debug_embeddings/image_embeds.npy"))#.to(device)
    text_embeds = torch.from_numpy(np.load(f"/data/jiahao/{temp_feature_dir}/debug_embeddings/text_embeds.npy"))#.to(device)
    text_atts = torch.from_numpy(np.load(f"/data/jiahao/{temp_feature_dir}/debug_embeddings/text_atts.npy"))#.to(device)

    sims_test_result = mAP(sims_matrix_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    table.add_row([
        -999, sims_test_result['R1'], sims_test_result['R5'], sims_test_result['R10'], sims_test_result['mAP'], sims_test_result['mINP']
    ])
    print("### Zero-Shot ITC Score: ")
    print(table)


    labels = test_loader.dataset.g_pids, test_loader.dataset.q_pids #TODO


    score_test_t2i = evaluation_itm(
        model,
        device, config, args,
        sims_matrix_t2i, image_embeds, text_embeds, text_atts
    )
    test_result = mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    table.add_row([
        -999, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
    ])
    print("### Zero-Shot ITM Score: ")
    print(table)

    ## CUHK
    # table.add_row([-999, 70.760, 87.378, 92.268, 63.849, 48.174])
    # table.add_row([-999, 76.170, 89.457, 93.600, 65.570, 47.372])
    # print("### Zero-Shot Score: ")
    # print(table)
    # ### Zero-Shot ITM Score: CUHK_finetuned
    # # +-------+--------+--------+--------+--------+--------+
    # # | epoch |   R1   |   R5   |  R10   |  mAP   |  mINP  |
    # # +-------+--------+--------+--------+--------+--------+
    # # |  -999 | 70.760 | 87.378 | 92.268 | 63.849 | 48.174 |
    # # |  -999 | 76.170 | 89.457 | 93.600 | 65.570 | 47.372 |
    # # +-------+--------+--------+--------+--------+--------+
    # ### Zero-Shot ITM Score: CUHK_pretrained
    # +-------+-------+--------+--------+-------+-------+
    # | epoch |   R1  |   R5   |  R10   |  mAP  |  mINP |
    # +-------+-------+--------+--------+-------+-------+
    # |  -999 | 5.133 | 11.127 | 15.156 | 4.918 | 2.297 |
    # |  -999 | 3.509 | 6.725  | 8.869  | 3.266 | 1.505 |

    # ### Zero-Shot ITM Score: ICFG_finetuned
    # +-------+--------+--------+--------+--------+--------+
    # | epoch |   R1   |   R5   |  R10   |  mAP   |  mINP  |
    # +-------+--------+--------+--------+--------+--------+
    # |  -999 | 62.999 | 79.484 | 85.041 | 40.410 | 10.184 |
    # |  -999 | 68.188 | 82.885 | 87.500 | 40.507 | 9.104  |
    # +-------+--------+--------+--------+--------+--------+

    # ### Zero-Shot ITM Score: RSTP_finetuned
    # +-------+--------+--------+--------+--------+--------+
    # | epoch |   R1   |   R5   |  R10   |  mAP   |  mINP  |
    # +-------+--------+--------+--------+--------+--------+
    # |  -999 | 59.000 | 79.650 | 88.500 | 47.231 | 25.876 |
    # |  -999 | 66.450 | 85.600 | 90.600 | 50.642 | 25.526 |
    # +-------+--------+--------+--------+--------+--------+


    if args.tta:
        print("### TTA:")


        print("### Compute ITC Uncertainty")
        recall_types, ss_idxs_list, uncertaintys_list, proba_top1_sim_list, proba_inversed_sim_list  = preprocess_tta_coefficients(config, sims_matrix_t2i)
        #cuhk_finetuned 6156->6156 #cuhk_pretrained 6156->? #icfg_finetuned 19848->? #rstp_finetuned 2000->?

        print("### Creating tta dataset")
        tta_dataset = create_tta_dataset(
            config,
            sims_matrix_t2i.cpu(),
            image_embeds.cpu(),
            text_embeds.cpu(),
            text_atts.cpu(),
            recall_types,
            ss_idxs_list,
            uncertaintys_list,
            proba_top1_sim_list,
            proba_inversed_sim_list,

        )
        print(f"     tta_dataset: {len(tta_dataset)}")
        # sample = next(iter(tta_dataset))
        # print(sample)

        print("### Creating tta dataloader")
        tta_loader = create_tta_loader(
            [tta_dataset],
            batch_size=[config['batch_size_tta']],
            num_workers=[4],
            is_trains=[True],
            collate_fns=[None]
        )[0]
        print(f"     tta_loader: {len(tta_loader)}")
        # batch = next(iter(tta_loader))
        # print(batch)


        print("### Configure adapted weights")
        # arg_tm = utils.AttrDict(config['tta_model'])
        model = configure_tta_model(config, model)
        print("     TTA Dropout Modules: \r\n", [(n,m,m.training) for n,m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.training==True] )
        print("     TTA Require Gradient Params: \r\n", [(n, p.shape) for n,p in model.named_parameters() if p.requires_grad] )
        print("     TTA Dropout Modules Number: \r\n", sum([ 1 for n,m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.training==True ]) )
        print("     TTA Require Gradient Params Sum: \r\n", sum(p.numel() for p in model.parameters() if p.requires_grad) )
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_tta_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil( len(tta_dataset) / config['batch_size_tta'] )
        lr_scheduler = create_tta_scheduler(arg_sche, optimizer)
        scaler = GradScaler()  # bf16


        print("### Start ITM Test Time Adaptation")
        start_time = time.time()
        best = 0
        best_epoch = 0
        best_logs = {}
        max_epoch = config['schedular']['epochs']
        for epoch in range(0, max_epoch):

            train_stats = test_time_adapt_itm(model, optimizer, scaler, epoch, device, lr_scheduler, config, tta_loader)

            if config.get('is_image_augmentation', False) == True:
                sims_matrix_t2i, image_embeds, text_embeds, text_atts = evaluation_itc(model, test_loader, tokenizer, device, config)
            score_test_t2i = evaluation_itm(
                model,
                device, config, args,
                sims_matrix_t2i, image_embeds, text_embeds, text_atts
            )

            test_result = mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
            table.add_row([
                epoch, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
            ])
            print("### TTA ITM Score: ")
            print(table)

            logs = {'epo': epoch}
            for k, v in test_result.items():
                logs[k] = np.around(v, 3)
            for k, v in train_stats.items():
                logs[k] = float(v)
            print('     logs: ', logs)

            for k, v in logs.items():
                logs[k] = str(v)
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(logs) + "\n")

            result = test_result['R1']
            if result > best:
                # save_obj = {'model': model.state_dict(), 'config': config, }
                # torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                best = result
                best_epoch = epoch
                best_logs = logs

            # del sims_matrix_t2i, image_embeds, text_embeds, text_atts
            torch.cuda.empty_cache()

        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f"best epoch {best_epoch} : {best_logs}")
        print(f"### best epoch {best_epoch} : {best_logs}")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--bs', default=0, type=int, help="mini batch size")
    parser.add_argument('--epo', default=0, type=int, help="epoch")
    parser.add_argument('--lr', default=0.0, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml = YAML(typ='rt')
    config = yaml.load(open(args.config, 'r'))
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
