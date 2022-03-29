from glob import glob


for xml_ann in glob('/home/denis/tf_cars_detection/workspace/training/images/Car_Bus_Truck/test/*.xml') :
    print(xml_ann)
    with open(xml_ann, 'r') as f:
        data = f.read()
    with open(xml_ann, 'w') as f:
        # data = data.replace('/home/denis/Car_Bus_Truck/', '/home/denis/tf_cars_detection/workspace/training/images/Car_Bus_Truck/test/')
        # data = data.replace('Car_Bus_Truck', 'test')
        data = data.replace('test/test', 'test')
        f.write(data)

for xml_ann in glob('/home/denis/tf_cars_detection/workspace/training/images/Car_Bus_Truck/train/*.xml'):
    print(xml_ann)
    with open(xml_ann, 'r+') as f:
        data = f.read()
    with open(xml_ann, 'w') as f:
        # data = data.replace('/home/denis/Car_Bus_Truck/', '/home/denis/tf_cars_detection/workspace/training/images/Car_Bus_Truck/train/')
        # data = data.replace('Car_Bus_Truck', 'train')
        data = data.replace('train/train', 'train')
        f.write(data)