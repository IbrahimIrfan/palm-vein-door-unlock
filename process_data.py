from processing import process_image

labels = ["ibrahim", "ayush"]

for label in labels:     
    for i in range(20):
        file_name = label + "_left_" + str(i) + ".jpg"
        print(file_name)
        process_image("data/raw/" + file_name, "data/processed/" + file_name)

    for i in range(20):
        file_name = label + "_right_" + str(i) + ".jpg"
        print(file_name)
        process_image("data/raw/" + file_name, "data/processed/" + file_name)
