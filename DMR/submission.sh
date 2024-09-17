INPUT_FILE=$1
OUTPUT_FILE=$2

python predict.py \
--input_file $INPUT_FILE \
--output_file $OUTPUT_FILE \
--config_file models/baseline.pkl