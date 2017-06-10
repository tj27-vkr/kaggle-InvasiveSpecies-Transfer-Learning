python ../TensorFlow/tensorflow/tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=tf_files/bottlenecks \
--how_many_training_steps 4000 \
--model_dir=tf_files/inception \
--output_graph=tf_files/retrained_graph.pb \
--output_labels=tf_files/retrained_labels.txt \
--summaries_dir=tf_files/logs \
--image_dir train