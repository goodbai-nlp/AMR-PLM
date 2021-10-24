
for cate in 32 64 128 256 512
do
    echo processing few-shot $cate
    datadir=AMR17-${cate}ins
    if [ ! -d $datadir ];then
        mkdir $datadir
    else
        echo $datadir exists
    fi
    python construct_few_shot.py AMR17-full $datadir $cate
    cp AMR17-full/test* $datadir/
done