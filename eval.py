import os
import argparse
import configparser

import chokepoint_eval as cp_eval
from utils import metric_util

parser = argparse.ArgumentParser(description='Evaluation on ChokePoint dataset')
parser.add_argument('--config', metavar='PATH',
                    help='path to evaluation configuration path', default='./config/eval.config')

if __name__ == "__main__":

    args = parser.parse_args()

    PATH_TO_CONFIG = args.config

    config = configparser.ConfigParser()
    config.read(PATH_TO_CONFIG)

    PATH_TO_OUTPUT = config['Usage']['output_path']

    # Element list to Evaluate
    dataset_list = config['Evaluation Config']['dataset'].split('\n')
    # Path to frozen detection graph. This is the actual model that is used for the object detection.

    model_list_raw = config['Evaluation Config']['model_list'].split('\n')
    model_list = []
    for model_data_raw in model_list_raw:
        model_data = model_data_raw.split(',')
        model_list.append({'model name': model_data[0], 'model path': model_data[1]})

    threshold_list = config['Evaluation Config']['threshold_list'].split('\n')
    threshold_list = [float(i) for i in threshold_list]


    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = config['Evaluation Config']['label_map_path']
    NUM_CLASSES = int(config['Evaluation Config']['num_class'])

    for dataset in dataset_list:

        #########################
        # Chokepoint Evaluation #
        #########################
        if dataset == 'Chokepoint':

            PATH_TO_DATA = config['Chokepoint']['dataset_path']
            video_list_raw = config['Chokepoint']['video_list'].split('\n')

            # Extract frame path and label path
            video_list = []
            for video_name in video_list_raw:
                row = config['Chokepoint file path'][video_name]
                video_list.append(row.split(','))

            model_tag = 1

            for threshold in threshold_list:
                print('\n##################\nThreshold {}\n##################\n'.format(threshold))
                for model in model_list:
                    print('\n##################\nModel {}\n##################\n'.format(model['model name']))

                    metrics = metric_util.frame_metric()

                    output_path = PATH_TO_OUTPUT + 'thr{}/{}/'.format(threshold, model['model name'])
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    for video in video_list:

                        video_metrics = cp_eval.chokepoint_eval(model['model path'], NUM_CLASSES, PATH_TO_DATA, PATH_TO_LABELS, video, threshold, output_path)

                        metrics.add_data({'frame_number': video,
                                          'false_detection': video_metrics['total FP'],
                                          'true_detection': video_metrics['total TP'],
                                          'miss_detection': video_metrics['total FN'],
                                          'elapsed_time': video_metrics['Mean elapsed time']})

                        # END VIDEO LOOP
                    #END MODEL LOOP

                    print('\n Overall results:')
                    metrics.print_result()
                    metrics.write_to_csv('overall_metrics.csv', output_path)

                #END THRESHOLD LOOP
            print('\nBenchmarking completed.')