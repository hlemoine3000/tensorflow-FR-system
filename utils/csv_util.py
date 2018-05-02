import os
import csv

def write_csv(file_name, output_path, data, header):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(output_path + file_name, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        # write header
        writer.writerow(header)

        # write data
        [writer.writerow(x) for x in data]

    return 0