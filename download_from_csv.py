import csv

with open('content.csv','r', newline='') as csvfile:
    r = csv.DictReader(csvfile)
    for row in r:
        url = row['视频地址']
        