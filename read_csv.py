import csv
import os
label_name = ['skirt_length_labels', 'coat_length_labels', 'collar_design_labels', 'lapel_design_labels',
              'neck_design_labels', 'neckline_design_labels', 'pant_length_labels', 'sleeve_length_labels']
skirt_length_values = ['Invisible', 'Short Length', 'Knee Length', 'Midi Length', 'Ankle Length', 'Floor Length']
skirt_length_dir_name = ['Invisible', 'ShortLength', 'KneeLength', 'MidiLength', 'AnkleLength', 'FloorLength']
train_dir = 'skirt/data/train/'
validation_dir = 'skirt/data/validation/'
image_dir = '/Users/shaoqing/FashonAI/dataset/base/'

if __name__ == '__main__':

    # for element in skirt_length_dir_name:
    #     os.system('mkdir '+train_dir+element)
    #     os.system('mkdir ' + validation_dir + element)

    csv_file = csv.reader(open('/Users/shaoqing/FashonAI/dataset/base/Annotations/label.csv'))
    for element in csv_file:
        if element[1] == 'skirt_length_labels':
            os.system('cp ' + image_dir + element[0] + ' '+train_dir +
                      skirt_length_dir_name[element[2].index('y')])
    # pass