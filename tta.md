# TTA records


# baseline score
**results in APTM paper**
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
3. python3 run.py --task "itr_icfg" --evaluate --dist "gpu1" --output_dir "output/ft_icfg/test" --checkpoint "checkpoints/ft_icfg/checkpoint_best.pth"
    +------+--------+--------+--------+--------+-------+
    | task |   R1   |   R5   |  R10   |  mAP   |  mINP |
    +------+--------+--------+--------+--------+-------+
    | t2i  | 68.183 | 82.885 | 87.500 | 40.507 | 9.104 |
    +------+--------+--------+--------+--------+-------+
4. python3 run.py --task "itr_pa100k" --evaluate --dist "gpu0" --output_dir "output/ft_pa100k/test" --checkpoint "checkpoints/ft_pa100k/checkpoint_best.pth"
    {'label_mA': 0.8235, 'ins_acc': 0.8024, 'ins_prec': 0.8853, 'ins_rec': 0.8769, 'ins_f1': 0.8811}
5. python3 run.py --task "itr_gene" --dist "f4" --output_dir "output/pretrained"
    仅用于pretrain
6. python3 run.py --task "itr_cuhk" --dist "f4" --output_dir "output/ft_cuhk" --checkpoint "output/pretrained/checkpoint_31.pth"
    仅用于finetuned


# command
- tta: run.py
    python3 run.py --task "itr_cuhk" --tta --dist "gpu0" --output_dir "output/ft_cuhk/tta" --checkpoint "checkpoints/ft_cuhk/checkpoint_best.pth"

- tta: tta.py
    <!-- CUDA_VISIBLE_DEVICES=1 python3 tta.py --tta --task tta --config configs/tta.yaml  --output_dir output/tta/2025081715503  --checkpoint checkpoint/cmp.pth --bs 32 --epo 10 --lr 0.0001 --seed 42 -->


# exp

## tta_debug
    <!-- python3 run.py --task "tta_debug" --tta --checkpoint "checkpoint/cmp.pth" --bs 1 --epo 10 --lr 1e-4 --seed 42 -->
    <!-- CUDA_VISIBLE_DEVICES=1 python3 tta.py --config configs/tta_debug.yaml --task tta_debug --output_dir output/tta_debug/2025081715503 --checkpoint checkpoint/cmp.pth --bs 3 --epo 10 --lr 0.0001 --seed 42 --tta -->

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

