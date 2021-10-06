for cate in train dev test
do
	echo "processing $cate..."
	python convert_to_jsonl.py $cate.src $cate.tgt $cate.jsonl
done
