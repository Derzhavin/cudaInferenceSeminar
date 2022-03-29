
import os

MODEL_NAME = 'my_ssd_resnet101_v1_fpn_640x640_coco17_tpu-8'

PATH_TO_CKPT = './training/exported-models/my_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint'
PATH_TO_CFG = './training/exported-models/my_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config'
PATH_TO_LABELS = './training/annotations/label_map.pbtxt'
CAMERA_SRC = '/media/denis/Seagate3/Projects/cudaInferenceSeminar/data/cars1.mp4'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import cv2

cap = cv2.VideoCapture(CAMERA_SRC, cv2.CAP_GSTREAMER)
resolution = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(resolution)

import numpy as np

while True:
    ret, image_np = cap.read()

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    detection_boxes_np = detections['detection_boxes'][0].numpy()
    detection_classes_np = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)
    detection_scores_np = detections['detection_scores'][0].numpy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detection_boxes_np,
          detection_classes_np,
          detection_scores_np,
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.40,
          agnostic_mode=False)

    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (1920, 1080)))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()