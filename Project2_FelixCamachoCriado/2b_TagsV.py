#import libraries
import xml.etree.cElementTree as ET
import pprint
from collections import defaultdict
import re
from pprint import pprint

file_run = 'OSM_XLM_Madrid_City_Map.osm'
#file_run = sample_Madrid.osm

#definition of criteria for the count
lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

d1 = defaultdict(int)

# Counter of tags
def key_count(d1, element):
    if element.tag == 'tag':
        d1[element.attrib['k']] += 1
    return d1


def process_map(filename):
    for _, element in ET.iterparse(filename):
        key_count(d1, element)
    return d1

def top_tags(d, limit_=50000):
    l1 = []
    for key in d:
        if d[key] > limit_:
            l1.append((key, d[key]))
    l1.sort(key=lambda n: n[1], reverse= True)
    
    return l1

#Adding limit of 50000 in order to show top tags

process_map(file_run)
pprint(top_tags(d1))