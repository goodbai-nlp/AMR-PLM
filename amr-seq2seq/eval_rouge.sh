python -m rouge_score.rouge \
	--target_filepattern=$1 \
	--prediction_filepattern=$2 \
	--output_filename=$3 \
	--use_stemmer=true
