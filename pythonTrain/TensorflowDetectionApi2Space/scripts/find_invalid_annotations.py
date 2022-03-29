import xml.etree.ElementTree as ET
from glob import glob


for xml_ann in glob('./training/images/test/*.xml') + glob('./training/images/train/*.xml'):
    tree = ET.parse(xml_ann)
    root = tree.getroot()
    for bndbox in root.iter('bndbox'):
        xmin = float(bndbox.findall('xmin')[0].text)
        xmax = float(bndbox.findall('xmax')[0].text)
        ymin = float(bndbox.findall('ymin')[0].text)
        ymax = float(bndbox.findall('ymax')[0].text)

        if xmin >= xmax:
            print(f'{xml_ann}, xmin >= xmax, {xmin}, {xmax}')

        if ymin >= ymax:
            print(f'{xml_ann}, ymin >= ymax, {ymin}, {ymax}')

    for name in root.iter('name'):
        if name.text not in ['Car', 'Bus', 'Truck']:
            print(xml_ann)
