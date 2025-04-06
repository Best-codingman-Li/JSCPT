#!/bin/bash
###
 # @Author: Guosy_wxy 1579528809@qq.com
 # @Date: 2024-06-20 03:27:58
 # @LastEditors: Guosy_wxy 1579528809@qq.com
 # @LastEditTime: 2025-01-23 22:24:30
 # @FilePath: /LCY/Prompt/mvlpt-master/scripts/mvlpt/main_mt_coopdata_cut.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
#./main_mt_coopdata_cut.sh CTPS vit_b16 16 6 1 
#./main_mt_coopdata_cut.sh CTPS vit_b16 20 6 6 

TRAINER=$1

output_dir=output

root=coop_data

# DATASET=$1 # ['hateful-memes', 'cifar-10', 'mnist', 'oxford-flower-102', 'oxford-iiit-pets', 'resisc45_clip', 'country211', 'food-101', 'stanford-cars', 'fgvc-aircraft-2013b-variants102', 'caltech-101', 'dtd', 'voc-2007-classification', 'cifar-100', 'patch-camelyon', 'rendered-sst2', 'gtsrb', 'eurosat_clip', 'fer-2013', 'kitti-distance']
CFG=$2  # config file
NCTX=$3  # number of context tokens
SHOTS=$4  # number of shots (5, 20, 50)

# DATASET="Caltech101,Food101,StanfordCars,OxfordPets,OxfordFlowers,FGVCAircraft,SUN397,DescribableTextures,EuroSAT,UCF101"
DATASET="Digit5,VisDA17,Office31,OfficeHome" # SUN397,UCF101,Caltech101#DescribableTextures,EuroSAT #,SUN397,Food101,OxfordFlowers,Caltech101 ###Digit5,VisDA17,Office31,OfficeHome### DomainNet,miniDomainNet,PACS,VLCS

# for SEED in 1 2 3
# for SEED in 1
for SEED in $5
do
    DIR=$output_dir/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    # if [ -d "$DIR" ]; then
    #     echo "Oops! The results exist at ${DIR} (so skip this job)"
    # else
    if [ $TRAINER = "MMAP" ]; then
       CUDA_VISIBLE_DEVICES=7 python3 ../train.py \
        --root $root \
        --seed ${SEED} \
        --trainer MMAP \
        --config-file configs/trainers/MVLPT/${CFG}.yaml \
        --output-dir ${DIR} \
        --dataset ${DATASET} \
        --shots ${SHOTS} \
        --dataset-coop \
        --multi-task \
        TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
        TRAINER.MVLPT.COOP.N_CTX ${NCTX} \
        TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
        TRAINER.MVLPT.COOP.CSC False \
        TEST.NO_TEST False \
		TEST.FINAL_MODEL "best_val" \
        TRAINER.CUT_CONTEXTLEN True
    elif [ $TRAINER = "CTPS" ]; then
        CUDA_VISIBLE_DEVICES=4,5,6,7 python3 ../ctps_train.py \
        --root $root \
        --seed ${SEED} \
        --trainer CTPS \
        --config-file configs/trainers/MVLPT/${CFG}.yaml \
        --output-dir ${DIR} \
        --dataset ${DATASET} \
        --shots ${SHOTS} \
        --dataset-coop \
        --multi-task \
        TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
        TRAINER.MVLPT.COOP.N_CTX ${NCTX} \
        TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
        TRAINER.MVLPT.COOP.CSC False \
        TEST.NO_TEST False \
		TEST.FINAL_MODEL "best_val" \
        TRAINER.CUT_CONTEXTLEN True
    elif [ $TRAINER = "TPCS_BIOP" ]; then
        CUDA_VISIBLE_DEVICES=4,5 python3 ../tpcs_biop_train.py \
        --root $root \
        --seed ${SEED} \
        --trainer MVLPT \
        --config-file configs/trainers/MVLPT/${CFG}.yaml \
        --output-dir ${DIR} \
        --dataset ${DATASET} \
        --shots ${SHOTS} \
        --dataset-coop \
        --multi-task \
        TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
        TRAINER.MVLPT.COOP.N_CTX 0 \
        TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
        TRAINER.MVLPT.COOP.CSC False \
        TEST.NO_TEST False \
		TEST.FINAL_MODEL "best_val" \
        TRAINER.CUT_CONTEXTLEN True
    elif  [ $TRAINER = "VPT" ]; then
        CUDA_VISIBLE_DEVICES=3 python3 ../train.py \
         --root $root \
         --seed ${SEED} \
         --trainer MVLPT \
         --config-file configs/trainers/MVLPT/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --multi-task \
         TRAINER.MVLPT.VPT.N_CTX ${NCTX} \
         TRAINER.MVLPT.COOP.N_CTX 0 \
         TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
         TRAINER.MVLPT.COOP.CSC False \
         TEST.NO_TEST False \
         TEST.FINAL_MODEL "best_val"
    elif [ $TRAINER = "MaPLe" ]; then
        CUDA_VISIBLE_DEVICES=3 python3 ../train.py \
         --root $root \
         --seed ${SEED} \
         --trainer ${TRAINER} \
         --config-file configs/trainers/MaPLe/${CFG}.yaml \
         --output-dir ${DIR} \
         --dataset ${DATASET} \
         --shots ${SHOTS} \
         --dataset-coop \
         --multi-task \
        TRAINER.MVLPT.VPT.N_CTX 0 \
        TRAINER.MVLPT.COOP.N_CTX ${NCTX} \
        TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
        TRAINER.MVLPT.COOP.CSC False \
        TEST.NO_TEST False \
		TEST.FINAL_MODEL "best_val" \
    else 
        CUDA_VISIBLE_DEVICES=7 python3 ../train.py \
        --root $root \
        --seed ${SEED} \
        --trainer MVLPT \
        --config-file configs/trainers/MVLPT/${CFG}.yaml \
        --output-dir ${DIR} \
        --dataset ${DATASET} \
        --shots ${SHOTS} \
        --dataset-coop \
        --multi-task \
        TRAINER.MVLPT.VPT.N_CTX 0 \
        TRAINER.MVLPT.COOP.N_CTX ${NCTX} \
        TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION 'middle' \
        TRAINER.MVLPT.COOP.CSC False \
        TEST.NO_TEST False \
		TEST.FINAL_MODEL "best_val" \
        TRAINER.CUT_CONTEXTLEN True
    fi
done
