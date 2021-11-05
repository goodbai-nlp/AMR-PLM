# !/bin/bash
# cate=AMR1.0
# for cate in AMR1.0 AMR2.0 AMR3.0
# do
#     echo "Preprocessing $cate ..."
#     mkdir -p ../data/$cate
#     CUDA_VISIBLE_DEVICES=0 python -u bin/preprocess.py --config configs/config-$cate.yaml --direction amr 2>&1 | tee preprocess-$cate.log 
# done

# cate=Giga
# echo "Preprocessing $cate ..."
# mkdir -p ../data/$cate
# CUDA_VISIBLE_DEVICES=0 python -u bin/preprocess.py --config configs/config-$cate.yaml --direction amr 2>&1 | tee preprocess-$cate.log 

cate=TLP
echo "Preprocessing $cate ..."
mkdir -p ../data/$cate
CUDA_VISIBLE_DEVICES=0 python -u bin/preprocess.py --config configs/config-$cate.yaml --direction amr 2>&1 | tee preprocess-$cate.log 

# cate=New3
# echo "Preprocessing $cate ..."
# mkdir -p ../data/$cate
# CUDA_VISIBLE_DEVICES=0 python -u bin/preprocess.py --config configs/config-$cate.yaml --direction amr 2>&1 | tee preprocess-$cate.log 

# cate=Bio
# echo "Preprocessing $cate ..."
# mkdir -p ../data/$cate
# CUDA_VISIBLE_DEVICES=0 python -u bin/preprocess.py --config configs/config-$cate.yaml --direction amr 2>&1 | tee preprocess-$cate.log 

cate=Taskdata
echo "Preprocessing $cate ..."
mkdir -p ../data/$cate
CUDA_VISIBLE_DEVICES=0 python -u bin/preprocess.py --config configs/config-$cate.yaml --direction amr 2>&1 | tee preprocess-$cate.log 
