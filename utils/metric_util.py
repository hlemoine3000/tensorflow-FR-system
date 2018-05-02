
from utils import csv_util

def Average(lst):
    return sum(lst) / len(lst)

class frame_metric:

    def __init__(self):

        self.frame_number = []

        self.false_detection = []
        self.true_detection = []
        self.miss_detection = []

        self.elapsed_time = []

    def add_data(self, _frame_data):

        self.frame_number.append(_frame_data['frame_number'])
        self.false_detection.append(_frame_data['false_detection'])
        self.true_detection.append(_frame_data['true_detection'])
        self.miss_detection.append(_frame_data['miss_detection'])
        self.elapsed_time.append(_frame_data['elapsed_time'])

    def print_result(self):

        print('False detections {}'.format(sum(self.false_detection)))
        print('True detections {}'.format(sum(self.true_detection)))
        print('Miss detections {}'.format(sum(self.miss_detection)))

        print('Mean model elapsed time {} ms\n'.format(round(Average(self.elapsed_time) * 1000,2)))

    def get_final_metrics(self):
        return {'total FP': sum(self.false_detection),
                'total TP': sum(self.true_detection),
                'total FN': sum(self.miss_detection),
                'Mean elapsed time': Average(self.elapsed_time)}

    def write_to_csv(self, file_name, output_path):

        data = []

        header = ['total FP',
                  'total TP',
                  'total FN',
                  'Mean elapsed time',
                  'frame number',
                  'false detection',
                  'true detection',
                  'miss detection',
                  'elapsed time']

        # General metrics
        data.append([sum(self.false_detection)])
        data.append([sum(self.true_detection)])
        data.append([sum(self.miss_detection)])
        data.append([Average(self.elapsed_time)])

        # Raw metrics
        data.append(self.frame_number)
        data.append(self.false_detection)
        data.append(self.true_detection)
        data.append(self.miss_detection)
        data.append(self.elapsed_time)



        csv_util.write_csv(file_name, output_path, data, header)
        # with open(self.csv_file_path + self.csv_file_name, "w") as csv_file:
        #     writer = csv.writer(csv_file, delimiter=',')
        #     writer.writerow(self.epoch)
        #     writer.writerow(self.values_loss_train)
        #     writer.writerow(self.values_loss_val)
        #     writer.writerow(self.values_acc_train)
        #     writer.writerow(self.values_acc_val)

        return 0