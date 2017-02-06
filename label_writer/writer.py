
def write_label_predictions_to_file(relative_file_path, labels):
    output_file_name = relative_file_path
    output_file = open(output_file_name, "w")

    # parts of the id for the sumbission format
    # I must make each id like "nips4b_birds_testfile0001.wav_classnumber_1"
    front = "nips4b_birds_testfile"
    end = ".wav_classnumber_"

    header = "ID,Probability"
    output_file.write(header + "\n")

    number_of_records = len(labels)
    number_of_classes = len(labels[0])
    for file_number in range(1, number_of_records + 1):
        for class_number in range(1, number_of_classes + 1):
            test_file_name = front + "{0:04d}".format(file_number) + end + str(class_number)
            prob = "{0:0.1f}".format(labels[file_number-1][class_number-1])
            line = test_file_name + "," + prob
            output_file.write(line + "\n")

    output_file.close()
