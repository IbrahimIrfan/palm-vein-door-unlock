from processing import processImage

labels = ["angad", "anushka", "ayush", "cindy", "david", "edwin", "ibrahim", "jason", "jun", "justin", "nick", "samir", "thomas", "will"]

for label in labels:     
    for i in range(5):
        file_name = label + "_left_" + i + ".jpg"
        print file_name
        processImage("data/raw/" + file_name, "data/processed/" + file_name)

    for i in range(5):
        file_name = label + "_right_" + i + ".jpg"
        print file_name
        processImage("data/raw/" + file_name, "data/processed/" + file_name)