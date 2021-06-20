from os import listdir
from os.path import isfile, join
import json

root = "/home/mmplab603/桌面/AI CUP/new/"


for json_file in listdir(root + 'json'):
    print(json_file)
    with open(root + 'json/' + json_file) as file:
        data = json.load(file)
        with open(root + 'json/' + json_file.replace('.json', '.jpg.txt'), 'w') as gt_file:
            for shapes in data['shapes']:
                data = ','.join(str(int(x)) + ',' + str(int(y)) for x, y in shapes['points']) + ',null\n'
                gt_file.write(data)
                print(data)

# with open('test_list.txt', 'w') as file:
#     for i in range(1,101):
#         file.write('img_{}.jpg\n'.format(i))


