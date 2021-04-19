import os
import csv
img_id = 0

folder_path = '../cs766_21spring/dataset/img_SeaBirds'
y = 0 # 0 seabird
split = 0 
place = y
path_list = os.listdir(folder_path) 
with open("data/list_bird.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerow(['img_id','img_filename','y','split','place','place_filename'])
    for img_path in path_list:
        writer.writerow([ str(img_id) , folder_path + img_path, str(y), str(split), str(place), img_path])