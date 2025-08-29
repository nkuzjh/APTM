# TTA records


# baseline score
## results in APTM paper
1. python3 run.py --task "itr_cuhk" --evaluate --dist "gpu0" --output_dir "output/ft_cuhk/test" --checkpoint "checkpoints/ft_cuhk/checkpoint_best.pth"
    +------+--------+--------+--------+--------+--------+
    | task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
    +------+--------+--------+--------+--------+--------+
    | t2i  | 76.170 | 89.457 | 93.600 | 65.570 | 47.372 |
    +------+--------+--------+--------+--------+--------+
2. python3 run.py --task "itr_rstp" --evaluate --dist "gpu0" --output_dir "output/ft_rstp/test" --checkpoint "checkpoints/ft_rstp/checkpoint_best.pth"
    +------+--------+--------+--------+--------+--------+
    | task |   R1   |   R5   |  R10   |  mAP   |  mINP  |
    +------+--------+--------+--------+--------+--------+
    | t2i  | 66.450 | 85.600 | 90.600 | 50.642 | 25.526 |
    +------+--------+--------+--------+--------+--------+
3. python3 run.py --task "itr_icfg" --evaluate --dist "gpu0" --output_dir "output/ft_icfg/test" --checkpoint "checkpoints/ft_icfg/checkpoint_best.pth"
    +------+--------+--------+--------+--------+-------+
    | task |   R1   |   R5   |  R10   |  mAP   |  mINP |
    +------+--------+--------+--------+--------+-------+
    | t2i  | 68.183 | 82.885 | 87.500 | 40.507 | 9.104 |
    +------+--------+--------+--------+--------+-------+

## 不符合setting的任务
**evaluate pretrain -> transfer learning baseline分数太低**
1. python3 run.py --task "itr_cuhk" --evaluate --dist "gpu0" --output_dir "output/ft_cuhk/test_pretrain" --checkpoint "checkpoints/pretrained/checkpoint_31.pth"
    +------+-------+-------+-------+-------+-------+
    | task |   R1  |   R5  |  R10  |  mAP  |  mINP |
    +------+-------+-------+-------+-------+-------+
    | t2i  | 3.509 | 6.725 | 8.869 | 3.266 | 1.505 |
    +------+-------+-------+-------+-------+-------+
2. python3 run.py --task "itr_rstp" --evaluate --dist "gpu0" --output_dir "output/ft_rstp/test_pretrain" --checkpoint "checkpoints/pretrained/checkpoint_31.pth"
    +------+-------+--------+--------+-------+-------+
    | task |   R1  |   R5   |  R10   |  mAP  |  mINP |
    +------+-------+--------+--------+-------+-------+
    | t2i  | 7.200 | 14.600 | 20.600 | 4.992 | 1.957 |
    +------+-------+--------+--------+-------+-------+
3. python3 run.py --task "itr_icfg" --evaluate --dist "gpu0" --output_dir "output/ft_icfg/test_pretrain" --checkpoint "checkpoints/pretrained/checkpoint_31.pth"
    +------+-------+-------+-------+-------+-------+
    | task |   R1  |   R5  |  R10  |  mAP  |  mINP |
    +------+-------+-------+-------+-------+-------+
    | t2i  | 1.129 | 3.008 | 4.696 | 0.605 | 0.197 |
    +------+-------+-------+-------+-------+-------+
**evaluate itr_pa100k为i2t任务且模型结构不同**
1. python3 run.py --task "itr_pa100k" --evaluate --dist "gpu0" --output_dir "output/ft_pa100k/test" --checkpoint "checkpoints/ft_pa100k/checkpoint_best.pth"
   {'label_mA': 0.8235, 'ins_acc': 0.8024, 'ins_prec': 0.8853, 'ins_rec': 0.8769, 'ins_f1': 0.8811}
**train irt_gene为finetune任务**
2. python3 run.py --task "itr_gene" --dist "f4" --output_dir "output/pretrained"
    仅用于pretrain
**train itr_cuhk为finetune任务**
3. python3 run.py --task "itr_cuhk" --dist "f4" --output_dir "output/ft_cuhk" --checkpoint "output/pretrained/checkpoint_31.pth"
    仅用于finetuned


# command

- tta: run_tta.py
    python3 run_tta.py --task "itr_cuhk" --tta --config tta_configs/TTA_Retrieval_cuhk/exp_debug.yaml --output_dir "output/ft_cuhk/tta/exp_debug" --checkpoint "checkpoints/ft_cuhk/checkpoint_best.pth"

- tta: tta.py
    1. CUDA_VISIBLE_DEVICES=0 python3 tta.py --task itr_cuhk --config tta_configs/TTA_Retrieval_cuhk/exp_debug.yaml --output_dir output/ft_cuhk/tta/exp_debug --bs 3 --epo 10 --checkpoint checkpoints/ft_cuhk/checkpoint_best.pth --tta

    2. CUDA_VISIBLE_DEVICES=2 python3 tta.py --task itr_icfg --config tta_configs/TTA_Retrieval_icfg/exp_debug.yaml --output_dir output/ft_icfg/tta/exp_debug --bs 3 --epo 10 --checkpoint checkpoints/ft_icfg/checkpoint_best.pth --tta

    3. CUDA_VISIBLE_DEVICES=2 python3 tta.py --task itr_rstp --config tta_configs/TTA_Retrieval_rstp/exp_debug.yaml --output_dir output/ft_rstp/tta/exp_debug --bs 3 --epo 10 --checkpoint checkpoints/ft_rstp/checkpoint_best.pth --tta

- tta_pretrain: tta.py
    1. CUDA_VISIBLE_DEVICES=0 python3 tta.py --task itr_cuhk --config tta_configs/TTA_Retrieval_cuhk_pretrain/exp_debug.yaml --output_dir output/ft_cuhk/tta_pretrain/exp_debug --bs 3 --epo 10 --checkpoint checkpoints/pretrained/checkpoint_31.pth --tta

    2. CUDA_VISIBLE_DEVICES=2 python3 tta.py --task itr_icfg --config tta_configs/TTA_Retrieval_icfg_pretrain/exp_debug.yaml --output_dir output/ft_icfg/tta_pretrain/exp_debug --bs 3 --epo 10 --checkpoint checkpoints/pretrained/checkpoint_31.pth --tta

    3. CUDA_VISIBLE_DEVICES=2 python3 tta.py --task itr_rstp --config tta_configs/TTA_Retrieval_rstp_pretrain/exp_debug.yaml --output_dir output/ft_rstp/tta_pretrain/exp_debug --bs 3 --epo 10 --checkpoint checkpoints/pretrained/checkpoint_31.pth --tta


# exp

## exp0
**entropy**
- nohup python3 run.py --tta --task "tta_exp0"> logs/tta_exp0.log 2>&1 &
    0.0

    0.1

    0.2

    0.3

    0.4

    0.5

## exp1
**entropy + ss**
- nohup python3 run.py --tta --task "tta_exp1"> logs/tta_exp1.log 2>&1 &
    1.0

    1.1

    1.2

    1.3

    1.4

    1.5

## exp2
**entropy + ss + unc**

## exp3
**entropy + unc**

## exp4
**entropy + unc_temper_learn**

## exp5
**entropy + ss + unc_temper_learn**

## exp6
**entropy + pl**

## exp7
**entropy + ss + unc + pl**

## exp8
**entropy + iaug**

## exp9
**entropy + ss + unc + pl + iaug**
- 由于PAB CMP_exp8效果不好，不试验exp9的setting了

