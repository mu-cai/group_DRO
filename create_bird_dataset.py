import os
import csv
img_id = 0

big_folder_path = '../cs766_21spring/dataset/'
dataset_list = ['img_SeaBirds', 'img_Backyard Birds', 'test_Seabirds', 'test_Backyard Birds', 'val_seabirds', 'val_Backyard Birds']

index = 1
for dataset_name in dataset_list:
    folder_path = big_folder_path + dataset_name
    y = (index+1)%2  # 0 seabird
    split = (index-1)//2
    place = y
    print(folder_path, 'y=' , y, ' , split=' , split)
    path_list = os.listdir(folder_path) 
    with open("data/list_bird.csv","w") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(['img_id','img_filename','y','split','place','place_filename'])
        for img_path in path_list:
            writer.writerow([ str(img_id) , folder_path + img_path, str(y), str(split), str(place), img_path])
        img_id += 1
    index += 1