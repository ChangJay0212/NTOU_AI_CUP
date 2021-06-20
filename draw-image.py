import cv2
import os
import numpy as np

img_path='/home/mmplab603/桌面/9667/img'
txt_path='/home/mmplab603/桌面/9667/1'
save_path='/home/mmplab603/桌面/9667/csave'

# check_path='G:\\AI Cup\\DB\\res50\\check'
#check_path='/home/mmplab603/桌面/res50/check'

file=os.listdir(img_path)
#c = open(os.path.join(check_path,'check.txt'),'a')
for i in file:
    image = cv2.imread(os.path.join(img_path,i))
    f=open(os.path.join(txt_path,"res_"+i.replace('jpg','txt')),'r')

    lines = f.readlines()
    """
    if len(lines)<=1 or len(lines)>5:
        cv2.imwrite(os.path.join(check_path, i), image)
        c.write(i+'\n')
        """
    for line in lines:
        results = line.split(',')[:-1]
        # print(results)
        # results = list(map(int, str2))
        results =[int(float(i)) for i in (line.split(',')[0:-1])]
        pts = np.array(results, np.int32)
        pts = pts.reshape((-1,2))
        image = cv2.polylines(image, [pts], True, (0,0,255),2)
        # print(pts)
        # cv2.rectangle(image, (int(results[1]),int(results[2])), (int(results[5]), int(results[6])), (0, 0, 255), 2)
    f.close()
    print(os.path.join(save_path,i))
    cv2.imwrite(os.path.join(img_path,i), image)
c.close
