# Importing modules
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint
import csv
import codecs
import cerberus
from schema import schema

regex = re.compile(r'\b\S+\.?', re.IGNORECASE)

# after some auditing, these are the most common issues that can be fixed so each row which is refered to a specific kind of way in the dataset.

mapping_street = { 'A': 'Avenida','AV': 'Avenida', 'AV.': 'Avenida','AVDA.': 'Avenida', 
            'AVENIDA.': 'Avenida','Av': 'Avenida', 'Avda': 'Avenida','Avda.': 'Avenida',
            'Avenidad.': 'Avenida','Avenidad': 'Avenida', 'Av.' : 'Avenida','Avd.':'Avenida',
            'Autop.':'Autopista',
            'C/': 'Calle', 'Cl': 'Calle',
            'C.C.': 'Centro Comercial','C.C': 'Centro Comercial', 'C.c.':'Centro Comercial',
            'CR': 'Carretera','CRTA.': 'Carretera', 'CTRA.': 'Carretera','CTRA:': 'Carretera', 
            'Cr' : 'Carretera','Crta':'Carretera','CRTA:':'Carretera', 'Crta:':'Carretera',
            'Carretera/Carrera': 'Carretera','CRTA.':'Carretera','Crta.':'Carretera',
            'Carretera/carrera': 'Carretera','Carrterera': 'Carretera', 'Ctra.': 'Carretera',
            'Ctra': 'Carretera', 'crta': 'Carretera',
            'Pasage': 'Pasaje', 'Pz': 'Plaza', 'Pza':'Plaza',
            'Urb.':'Urbanizacion'
         }

#https://es.wikipedia.org/wiki/%C3%81rea_metropolitana_de_Madrid

#after some auditing, these are the most common issues that can be fixed so each row which is refered to a specific village or city will
#be group under a unique name.

mapping_city = { 
             '3':'',
             'Alcalá De Henares Madrid':'Alcala De Henares',
             'Alcalá De Henares (madrid)':'Alcala De Henares',
             'Alcorcón' : 'Alcorcon',
             'Barajas Madrid' : 'Barajas',
             'Collado Villalba':'Collado-Villalba',
             'Getafe (Madrid)':'Getafe',
             'Fuente El Saz Del Jarama' : 'Fuente El Saz De Jarama',
             'Las Rozas De Madrid':'Las Rozas','Rozas De Madrid':'Las Rozas',
             'Leganés':'Leganes',
             'Loeches (madrid)':'Loeches',
             'Madrd':'Madrid',
             'Majadahonda Madrid' : 'Majadahonda',
             'Móstoles' : 'Mostoles',
             'RIvas Vaciamadrid':'Rivas-Vaciamadrid','Rivas Vaciamadrid':'Rivas-Vaciamadrid',
             'Rivas Vaciamadrid3':'Rivas-Vaciamadrid','Rivas - Vaciamadrid':'Rivas-Vaciamadrid',
             'Daganzo De Arriba' :'Daganzo',
             'Paracuellos Del Jarama':'Paracuellos De Jarama',
             'Pozuelo De Alascon':'Pozuelo De Alarcon',
             'Pozuelo De Alarcón':'Pozuelo De Alarcon',
             'San Agustín Del Guadalix' : 'San Agustin de Guadalix',
             'San Sebastían De Los Reyes': 'San Sebastian De Los Reyes',
             'San Sebastian De Los Reyes' : 'San Sebastian De Los Reyes',
             'Torrejón De Ardoz' : 'Torrejon De Ardoz',
             'Villaviciosa De Odón' : 'Villaviciosa De Odon',
            }  

# This function is used for cleaning the street names and unify the nomenclature for the different cities and villages.
def update_name(name, mapping):
    """Add the name of the element to be updated and the mapping dict with the correct values
    Args: name(string): Street name or City Name in this case
          mapping(dict): Dict which include the know issue and the correct value to be updated.
    
    Returns:
          name(string): Correct name for each element
    """
    
    unwanted = [',']  # List of unwanted characters 
    el = ''                  
 
    #remove unwanted characters
    for i in range(len(name)):
        if name[i] not in unwanted:
            el = el + name[i]

    #Capitalize the first letter of each element and put to lower case the rest of letters
    low_name = el.lower()
    if ' ' in low_name:
        el = ''
        l = low_name.split(' ')
        for i in l:
            el = el + ' ' + i.capitalize()
    else:
        el = low_name.capitalize()

    #Match with mapping dict and in case it found some know issue/value, it replaces it for the correct form.
    k = mapping.keys()
    key_list = list(k)
    for abrev in key_list:
        if abrev in el.split():
            el = el.replace(abrev,mapping[abrev])

    return el

# Preparing for Database - SQL

#OSM_PATH = 'sample_Madrid.osm'
OSM_PATH = 'OSM_XLM_Madrid_City_Map.osm'
NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"


LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

SCHEMA = schema

NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']

def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []  # Handle secondary tags the same way for both node and way elements
    tags_way = []

    if element.tag == 'node':
        for i in node_attr_fields:
            node_attribs[i]=element.attrib[i]
        
        for secondary_tag in element.iter('tag'):
                tag_elements = {}
                tag_elements['id'] = element.attrib['id']
                if secondary_tag.attrib['k'] == 'addr:street':
                     result = update_name(secondary_tag.attrib['v'],mapping_street)
                     tag_elements['value'] = result
                elif secondary_tag.attrib['k'] == 'addr:city':
                     result = update_name(secondary_tag.attrib['v'],mapping_city)
                     tag_elements['value'] = result
                else:
                     tag_elements['value'] = secondary_tag.attrib['v']
    
    # get the match objects for problem and colon characters
                p = problem_chars.search(secondary_tag.attrib['k'])
                l = LOWER_COLON.search(secondary_tag.attrib['k'])
     # check if it has problem characters
                if p:
                 return
     # check if there are colons in the value of k
                elif l:
                 tag_elements['key'] = secondary_tag.attrib['k'].split(':',1)[1]
                 tag_elements['type'] = secondary_tag.attrib['k'].split(':',1)[0]
                else:
                 tag_elements['key'] = secondary_tag.attrib['k']
                 tag_elements['type'] = default_tag_type
                
                tags.append(tag_elements)
                
        return({'node': node_attribs, 'node_tags': tags})
 
    elif element.tag == 'way':
        
        for i in WAY_FIELDS:
            way_attribs[i]=element.attrib[i]
        
        for secondary_tag in element.iter('tag'):
                tag_elements = {}
                tag_elements['id'] = element.attrib['id']
                if secondary_tag.attrib['k'] == 'addr:street':
     #Here is when the function update is called and the data from the original .osm file are corrected for Streets
                     result = update_name(secondary_tag.attrib['v'],mapping_street)
                     tag_elements['value'] = result
     #Here is when the function update is called and the data from the original .osm file are corrected for Cities
                elif secondary_tag.attrib['k'] == 'addr:city':
                     result = update_name(secondary_tag.attrib['v'],mapping_city)
                     tag_elements['value'] = result
                else:
                  tag_elements['value'] = secondary_tag.attrib['v']

     # check for problematic characters and colons.
                p = problem_chars.search(secondary_tag.attrib['k'])
                l = LOWER_COLON.search(secondary_tag.attrib['k'])
                if p:
                 return

                elif l:
                 tag_elements['key'] = secondary_tag.attrib['k'].split(':',1)[1]
                 tag_elements['type'] = secondary_tag.attrib['k'].split(':',1)[0]
                else:
                 tag_elements['key'] = secondary_tag.attrib['k']
                 tag_elements['type'] = default_tag_type
                
                tags_way.append(tag_elements)
                    
        pos = 0
        for node in element.iter('nd'):
            waynd_dt = {}
            waynd_dt['id'] = element.attrib['id']
            waynd_dt['node_id'] = node.attrib['ref']
            waynd_dt['position'] = pos
            pos += 1
            way_nodes.append(waynd_dt)
            
        return({'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags_way})


# Helper Functions
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()

def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(iter(validator.errors.items()))
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_string = pprint.pformat(errors)
        
        raise Exception(message_string.format(field, error_string))


class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""
#made some modifications as I didn't want to handle binary values for Python 3.6
    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v if isinstance(v, str) else v) for k, v in row.items()
        })

#old function
#def writerow(self, row): 
#    super(UnicodeDictWriter, self).writerow({
#        k: (v.encode('utf-8') if isinstance(v, str) else v) for k, v in row.items()
#    })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)
            

def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w') as ways_file, \
        codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])

if __name__ == '__main__':
    process_map(OSM_PATH, validate=True)
    print('{} have been proccessed!'.format(OSM_PATH[:-4]))