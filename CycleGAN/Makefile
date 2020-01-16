# set hyperparameters here
BATCH_SIZE = 4
IMAGE_SIZE = 256
NGF = 64
X = line116
Y = pic

# for export_graph: default is the lastest checkpoint
CHECKPOINT_DIR = checkpoints/`ls checkpoints | tail -n 1`

# for inference
INPUT_IMG = input_sample.jpg
OUTPUT_IMG = output_sample.jpg
MODEL = pretrained/$(X)2$(Y).pb

# commands come here
download_data:
	bash download_dataset.sh $(X)2$(Y)

build_data:
	python3 build_data.py --X_input_dir=data/$(X)2$(Y)/train1 \
                              --Y_input_dir=data/$(X)2$(Y)/train2 \
                              --X_output_file=data/tfrecords/$(X).tfrecords \
                              --Y_output_file=data/tfrecords/$(Y).tfrecords

train:
	python3 train.py --batch_size=$(BATCH_SIZE) \
                         --load_model 20200115-0925 \
                         --image_size=$(IMAGE_SIZE) \
                         --ngf=$(NGF) \
                         --X=data/tfrecords/$(X).tfrecords \
                         --Y=data/tfrecords/$(Y).tfrecords

export_graph:
	CUDA_VISIBLE_DEVICES=-1  python3 export_graph.py --checkpoint_dir=$(CHECKPOINT_DIR) \
                                --XtoY_model=$(X)2$(Y).pb \
                                --YtoX_model=$(Y)2$(X).pb \
                                --image_size=$(IMAGE_SIZE)

inference:
	CUDA_VISIBLE_DEVICES=-1 python3 inference.py --model=$(MODEL)\
                             --input=$(INPUT_IMG) \
                             --output=$(OUTPUT_IMG) \
                             --image_size=$(IMAGE_SIZE)

tensorboard:
	tensorboard --logdir=$(CHECKPOINT_DIR)
