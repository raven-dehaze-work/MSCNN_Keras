"""
match trans and hazy name so that can be used to train
the match result will be save into trans_hazy_pair_file_name.json
"""
import os
import json

# testing dir where locate the training datas
read_dir1 = './trans'
read_dir2 = './hazy'

# load file names
trans_names = [file for root, dirs, file in os.walk(read_dir1)][0]
hazy_names = [file for root,dirs,file in os.walk(read_dir2)][0]
# get the number of total files
file_nums = len(trans_names)

# remove suffix
trans_without_suffix = list(map(
    lambda x: x.split('.')[0]+'_',      # add '_' char to match hazy_name
    trans_names
))

# declare pairs var to save trans and hazy which are matched
pairs = {}
pairs['trans'] = ['' for i in range(file_nums)]
pairs['hazy'] = ['' for i in range(file_nums)]

# design a function to match trans_names and hazy_names to form a pair so that can be used to train the network
count = 0
for idx,trans_name in enumerate(trans_without_suffix):
    for hazy_name in hazy_names:
        if trans_name in hazy_name:
            count+=1
            print('trans %s hazy %s'%(trans_names[idx],hazy_names[idx]))
            pairs['trans'][idx] = trans_names[idx]
            pairs['hazy'][idx] = hazy_names
            break

# print match pair info
print('the number of matched paired is %d, and total file number is %d' % (count,file_nums))

# save pair dict info file as json format
with open('trans_hazy_pair_file_name.json','w') as f:
    json.dump(pairs,f)

#print(pairs)

