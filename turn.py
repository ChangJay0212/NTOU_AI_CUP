import os
import csv
import operator

path='/home/mmplab603/下載/DB-20210611T085515Z-001/DB/AICUP_result'


list_path=os.listdir(path)
k= open("/home/mmplab603/下載/DB-20210611T085515Z-001/DB/outputs.csv",'a')
for file in list_path:
    print(file)
    f = open(os.path.join(path,file), 'r')
    for line in f.readlines():
        k.write(file.replace("res_img_","").replace(".txt","")+","+line)
k.close()

#reader = csv.reader(open("/home/mmplab603/下載/DB-20210611T085515Z-001/DB/output.csv"))
#sortedlist = sorted(reader, key=operator.itemgetter(0))
#print(sortedlist)
# 3 or 'n' depending upon which column you want to sort the data
#with open("sorted_file.csv", 'w') as f:
#    csv.writer(f).writerows(sortedlist)
