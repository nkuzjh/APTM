import os
import argparse


# Set it correctly for distributed training across nodes
# NNODES = 1  # e.g. 1/2/3/4
# NPROC_PER_NODE = 4  # e.g. 4 gpus
# MASTER_ADDR = '127.0.0.1'
# MASTER_PORT = 3000  # 0~65536
# NODE_RANK = 0  # e.g. 0/1/2

# print("NNODES, ", NNODES)
# print("NPROC_PER_NODE, ", NPROC_PER_NODE)
# print("MASTER_ADDR, ", MASTER_ADDR)
# print("MASTER_PORT, ", MASTER_PORT)
# print("NODE_RANK, ", NODE_RANK)


# def get_dist_launch(args):  # some examples
#     if args.dist == 'f4':
#         return "CUDA_VISIBLE_DEVICES=0,1,2,3 WORLD_SIZE=4 python3 -m torch.distributed.launch --nproc_per_node=4 " \
#                "--nnodes=1 --master_port={:}".format(MASTER_PORT)

#     elif args.dist == 'f2':
#         return "CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 python3 -m torch.distributed.launch --nproc_per_node=2 " \
#                "--nnodes=1 --master_port={:}".format(MASTER_PORT)

#     elif args.dist == 'l2':
#         return "CUDA_VISIBLE_DEVICES=2,3 WORLD_SIZE=2 python3 -m torch.distributed.launch --nproc_per_node=2 " \
#                "--nnodes=1 --master_port={:}".format(MASTER_PORT)

#     elif args.dist == 'f-0':
#         return "CUDA_VISIBLE_DEVICES=1,2,3 WORLD_SIZE=3 python3 -m torch.distributed.launch --nproc_per_node=3 " \
#                "--nnodes=1 "

#     elif args.dist == 'f-1':
#         return "CUDA_VISIBLE_DEVICES=0,2,3 WORLD_SIZE=3 python3 -m torch.distributed.launch --nproc_per_node=3 " \
#                "--nnodes=1 "

#     elif args.dist == 'f-2':
#         return "CUDA_VISIBLE_DEVICES=0,1,3 WORLD_SIZE=3 python3 -m torch.distributed.launch --nproc_per_node=3 " \
#                "--nnodes=1 "

#     elif args.dist == 'f-3':
#         return "CUDA_VISIBLE_DEVICES=0,1,2 WORLD_SIZE=3 python3 -m torch.distributed.launch --nproc_per_node=3 " \
#                "--nnodes=1 "

#     elif args.dist.startswith('gpu'):  # use one gpu, --dist "gpu0"
#         num = int(args.dist[3:])
#         assert 0 <= num <= 3
#         return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 python3 -m torch.distributed.launch --nproc_per_node=1 " \
#                "--nnodes=1 --master_port={:} ".format(num, MASTER_PORT)

#     else:
#         raise ValueError


def run_retrieval(args):
    # dist_launch = get_dist_launch(args)

    # os.system(f"{dist_launch} "
    #           f"--use_env Retrieval.py --config {args.config} "
    #           f"--task {args.task} --output_dir {args.output_dir} --bs {args.bs} --epo {args.epo} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")
    device_no=args.device_no
    os.system( f"CUDA_VISIBLE_DEVICES={device_no} python3 tta.py --config {args.config} --task {args.task} --output_dir {args.output_dir} --bs {args.bs} --epo {args.epo} --checkpoint {args.checkpoint} {'--tta' if args.tta else ''}" )


def run(args):
    # if args.task not in ['itr_gene']:
    #     assert os.path.exists(args.checkpoint)

    if args.task == 'itr_cuhk':
        assert os.path.exists("images/CUHK-PEDES")
        run_retrieval(args)

    elif args.task == 'itr_icfg':
        assert os.path.exists("images/ICFG-PEDES")
        run_retrieval(args)

    elif args.task == 'itr_rstp':
        assert os.path.exists("images/RSTPReid")
        run_retrieval(args)

    # elif args.task == 'itr_gene':
    #     assert os.path.exists("images/CUHK-PEDES")
    #     run_retrieval(args)

    # elif args.task == 'itr_pa100k':
    #     assert os.path.exists("images/PA100K")
    #     run_retrieval(args)

    else:
        raise NotImplementedError(f"task == {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--device_no', type=str, required=True, help="CUDA_VISIBLE_DEVICES")
    # parser.add_argument('--dist', type=str, required=True, help="see func get_dist_launch for details")
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus; ")
    parser.add_argument('--epo', default=-1, type=int, help="epoch")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--checkpoint', default='output/pretrain/checkpoint_31.pth', type=str, help="for fine-tuning")
    parser.add_argument('--output_dir', type=str, required=True, help='local path; ')
    parser.add_argument('--evaluate', action='store_true', help="evaluation on downstream tasks")
    parser.add_argument('--tta', action='store_true', help="test-time adapt on downstream tasks")
    parser.add_argument('--config', default='tta_configs/TTA_Retrieval_cuhk/exp_debug.yaml', type=str)
    args = parser.parse_args()

    # print(os.path.dirname(args.output_dir))
    # print(os.path.exists(os.path.dirname(args.output_dir)))
    # assert os.path.exists(os.path.dirname(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    run(args)



#####
##### results in APTM paper
#####
# python3 run.py --task "itr_cuhk" --evaluate --dist "gpu0" --output_dir "output/ft_cuhk/test" --checkpoint "checkpoints/ft_cuhk/checkpoint_best.pth"
# +------+--------+--------+--------+--------+--------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
# +------+--------+--------+--------+--------+--------+
# | t2i  | 76.170 | 89.457 | 93.600 | 65.570 | 47.372 |
# +------+--------+--------+--------+--------+--------+

# python3 run.py --task "itr_rstp" --evaluate --dist "gpu0" --output_dir "output/ft_rstp/test" --checkpoint "checkpoints/ft_rstp/checkpoint_best.pth"
# +------+--------+--------+--------+--------+--------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
# +------+--------+--------+--------+--------+--------+
# | t2i  | 66.450 | 85.600 | 90.600 | 50.642 | 25.526 |
# +------+--------+--------+--------+--------+--------+

# python3 run.py --task "itr_icfg" --evaluate --dist "gpu1" --output_dir "output/ft_icfg/test" --checkpoint "checkpoints/ft_icfg/checkpoint_best.pth"
# +------+--------+--------+--------+--------+-------+
# | task |   R1   |   R5   |  R10   |  mAP   |  mINP |
# +------+--------+--------+--------+--------+-------+
# | t2i  | 68.183 | 82.885 | 87.500 | 40.507 | 9.104 |
# +------+--------+--------+--------+--------+-------+

# python3 run.py --task "itr_pa100k" --evaluate --dist "gpu0" --output_dir "output/ft_pa100k/test" --checkpoint "checkpoints/ft_pa100k/checkpoint_best.pth"
# {'label_mA': 0.8235, 'ins_acc': 0.8024, 'ins_prec': 0.8853, 'ins_rec': 0.8769, 'ins_f1': 0.8811}


# python3 run.py --task "itr_gene" --dist "f4" --output_dir "output/pretrained"
# 仅用于pretrain
# python3 run.py --task "itr_cuhk" --dist "f4" --output_dir "output/ft_cuhk" --checkpoint "output/pretrained/checkpoint_31.pth"
# 仅用于finetuned

