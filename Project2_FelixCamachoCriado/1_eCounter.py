#import libraries
import xml.etree.cElementTree as ET
import pprint

#.OSM file include info downloaded from OpenStreetMap about Metropolitan area of Madrid
file_run = 'OSM_XLM_Madrid_City_Map.osm'
#file_run = sample_Madrid.osm

#function count_tag analyze the .osm file and count each type of element.
def count_tags(filename):
    tags={}
    for event, el in ET.iterparse(filename, events=("start",)):
        if el.tag in tags.keys():
            tags[el.tag] += 1
        else:
            tags[el.tag] = 1
    return tags
    

tags = count_tags(file_run)
sortedTags = sorted(tags.items(), key=lambda x: x[1], reverse=True)

pprint.pprint(sortedTags)
