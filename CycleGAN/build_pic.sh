#!/bin/sh

python build_data.py --X_input_dir=data/pic2line/train1 --Y_input_dir=data/pic2line/train2 \
	--X_output_file=data/tfrecords/pic.tfrecords --Y_output_file=data/tfrecords/lines.tfrecords



