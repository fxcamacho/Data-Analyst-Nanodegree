#import libraries
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

file_run = 'OSM_XLM_Madrid_City_Map.osm'
#file_run = 'sample_Madrid.osm'

regex = re.compile(r'\b\S+\.?', re.IGNORECASE)

#expected names in the dataset which represent the most typical kind of ways in Spanish.
expected_street = ["Calle","Avenida","Acceso","Aeropuerto", "Bulevar","Carretera","Cava","Campus","Cañada","Callejón","Carrera","Camino","Autovía",
"Corral","Cuesta","Corredera","Costanilla","Glorieta","Pasaje","Pasadizo","Parque","Paseo","Plaza","Polígono","Ronda",
"Rotonda","Prolongación","Travesía","Sector","Senda","Urbanización","Vía","Zona"] 

# Check if each value is matched with the regular expressions and if so and it is not included in the list of expected cities, then
#will be added to the key.

def audit_street(street_types, street_name): 
    m = regex.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected_street:
            street_types[street_type].add(street_name)

# Check if the element is a street name
def is_street_name(elem): 
    return (elem.attrib['k'] == "addr:street")

def audit(osmfile): 
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street(street_types, tag.attrib['v'])

    return street_types

pprint.pprint(dict(audit(file_run))) 
# print the existing street names, so this dict will show me the main issues that I will update and correct in the data cleaning phase.