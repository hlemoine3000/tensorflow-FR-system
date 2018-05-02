#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101
# code from https://github.com/yeephycho/tensorflow-face-detection.git

import sys
import os
import time
import numpy as np
import tensorflow as tf
import cv2
import argparse
import configparser
import xml.etree.ElementTree as ET

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_color as vis_util
from utils import metric_util

WHITE = (255, 255, 255)
BLUE = (255, 0, 0)

parser = argparse.ArgumentParser(description='Evaluation on ChokePoint dataset')
parser.add_argument('--config', metavar='PATH',
                    help='path to chokepoint configuration path', default='./config/chokepoint.config')

def is_point_in_box(box, point):
    ymin, xmin, ymax, xmax = box
    x, y = point

    if ((x > xmin) and (x < xmax) and (y > ymin) and (y < ymax)):
        return True
    else:
        return False

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def chokepoint_eval(model_path, num_classes, data_path, label_path, video_path, threshold=0.7, output_path='./res/'):

    print('Loading frozen graph at {}'.format(model_path))
    print('Loading label map at {}\n'.format(label_path))

    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)



    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(graph=detection_graph, config=config) as sess:

            # Empty video output
            out = None

            # Parse ground truth XML file
            path_to_frames = data_path + video_path[0]
            path_to_ground_truth = data_path + video_path[1]
            print('Reading frames at {}'.format(path_to_frames))
            print('Reading groundtruth at {}'.format(path_to_ground_truth))

            tree = ET.parse(path_to_ground_truth)
            root = tree.getroot()
            video_name = root.get('name')

            #Metrics definition

            metrics = metric_util.frame_metric()

            for frame in root:

                # Get frame eyes location
                frame_number = frame.get('number')
                if (int(frame_number) % 1000 == 0):
                    print('Frame: '+ frame_number)

                image_path = path_to_frames + frame_number + ".jpg"
                image = cv2.imread(image_path, 1)
                if (image is None):
                    print('Image not found at' + image_path)
                    break
                h, w, ch = image.shape

                if out is None:
                    # Define the codec and create VideoWriter object
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    video_out_path = output_path + video_name + "_out.mp4"

                    print('Creating video output at {}'.format(video_out_path))
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    out = cv2.VideoWriter(video_out_path, fourcc, 25.0, (w, h))

                # Get frame annotation
                if (frame.find('person') != None):
                    person_id = frame[0].get('id')

                    left_eye = (int(frame[0][0].get('x')), int(frame[0][0].get('y')))
                    right_eye = (int(frame[0][1].get('x')), int(frame[0][1].get('y')))

                    left_eye_norm = (left_eye[0] / w, left_eye[1] / h)
                    right_eye_norm = (right_eye[0] / w, right_eye[1] / h)

                    person_data = {'Person_ID': person_id,
                                   'Left_eye': left_eye,
                                   'Right_eye': right_eye,
                                   'Left_eye_norm': left_eye_norm,
                                   'Right_eye_norm': right_eye_norm}
                # Else, there is no person on that frame
                else:
                    left_eye = None
                    right_eye = None
                    person_data = None


                image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                start_time = time.time()
                (boxes, scores, classes, num_detections) = sess.run(
                  [boxes, scores, classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
                elapsed_time = time.time() - start_time
                #print('inference time cost: {}'.format(elapsed_time))

                # Visualization of the results of a detection.
                # vis_util.visualize_boxes_and_labels_on_image_array(
                #     # image_np,
                #     image,
                #     np.squeeze(boxes),
                #     np.squeeze(classes).astype(np.int32),
                #     np.squeeze(scores),
                #     category_index,
                #     use_normalized_coordinates=True,
                #     line_thickness=4)

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                # Metrics reset
                false_detection = 0
                true_detection = 0
                miss_detection = 0
                person_detected = False

                #Detection check loop
                for i in range(boxes.shape[0]):
                    if scores[i] > threshold:
                        # detection.append(boxes[i])

                        display_str = []
                        display_str.append('{}%'.format(int(100 * scores[i])))

                        ymin, xmin, ymax, xmax = boxes[i]


                        color = 'Blue'
                        if (person_data is not None):
                            if (is_point_in_box(boxes[i], person_data['Left_eye_norm'])
                                    and is_point_in_box(boxes[i], person_data['Right_eye_norm'])):
                                color = 'Green'
                                true_detection += 1
                                person_detected = True
                            else:
                                false_detection += 1
                        else:
                            false_detection += 1

                        vis_util.draw_bounding_box_on_image_array(
                            image,
                            ymin,
                            xmin,
                            ymax,
                            xmax,
                            color=color,
                            thickness=4,
                            display_str_list=display_str,
                            use_normalized_coordinates=True)
                # END DETECTION CHECK LOOP

                if (not person_detected and (person_data is not None)):
                    miss_detection += 1

                metrics.add_data({'frame_number': frame_number,
                                  'false_detection': false_detection,
                                  'true_detection' : true_detection,
                                  'miss_detection' : miss_detection,
                                  'elapsed_time' : elapsed_time})

                cv2.putText(image, "Frame: " + frame_number, (10, 30),
                            cv2.FONT_HERSHEY_TRIPLEX, 1, WHITE, 1, cv2.LINE_4)
                cv2.putText(image, "Detection time: %i ms" % (elapsed_time *1000), (10, 60),
                            cv2.FONT_HERSHEY_TRIPLEX, 1, WHITE, 1, cv2.LINE_4)


                if person_data is not(None):
                    cv2.line(image, person_data['Left_eye'], person_data['Left_eye'], BLUE, 5)
                    cv2.line(image, person_data['Right_eye'], person_data['Right_eye'], BLUE, 5)

                out.write(image)
                # END FRAMES LOOP

            print('\n' + video_name + ' results:')
            metrics.print_result()
            metrics.write_to_csv(video_name + '_metrics.csv', output_path)
            return metrics.get_final_metrics()
            # END SESSION
        # END DETECTION GRAPH
    # END MAIN

if __name__ == "__main__":

    args = parser.parse_args()

    PATH_TO_CONFIG = args.config

    cp_config = configparser.ConfigParser()
    cp_config.read(PATH_TO_CONFIG)

    PATH_TO_OUTPUT = cp_config['Usage']['output_path']

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = cp_config['Detection Model']['label_map_path']
    NUM_CLASSES = int(cp_config['Detection Model']['num_class'])
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = cp_config['Detection Model']['frozen_graph_path']

    threshold = float(cp_config['Detection Model']['threshold'])

    # Path to dataset
    PATH_TO_DATA = cp_config['Chokepoint Dataset']['dataset_path']
    eval_list_raw = cp_config['Chokepoint Dataset']['evalList'].split('\n')

    # Extract frame path and label path
    eval_list = []
    for video_name in eval_list_raw:
        video_path = cp_config['Chokepoint file path'][video_name].split(',')
        chokepoint_eval(PATH_TO_CKPT, NUM_CLASSES, PATH_TO_LABELS, video_path, threshold, PATH_TO_OUTPUT)