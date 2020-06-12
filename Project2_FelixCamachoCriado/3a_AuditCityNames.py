#import libraries
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

file_run = 'OSM_XLM_Madrid_City_Map.osm'
#file_run = 'sample_Madrid.osm' #The sample only includes info from one city, Madrid. So if you run the file with the sample the output will
#be empty as Madrid value is included in the expected city list.

regex = re.compile(r'\b\S+\.?', re.IGNORECASE)

expected_city = ["Madrid","Alcalá de Henares","Alcobendas","Alcorcón","Boadilla del Monte","Brunete","Colmenar Viejo","Coslada","Collado-Villalba",
"Fuenlabrada","Getafe","Leganés","Humanes de Madrid","Majadahonda","Daganzo", "Mejorada del Campo","Móstoles",
"Paracuellos de Jarama","Parla","Pinto","Pozuelo de Alarcon", "Las Rozas",
"Rivas-Vaciamadrid","Las Rozas de Madrid","San Fernando de Henares","San Sebastián de los Reyes","Torrejón de Ardoz","Tres Cantos",
"Velilla de San Antonio","Villanueva de la Cañada","Villanueva del Pardillo","Villaviciosa de Odón"]

#expected city names which should be part of this dataset. These names can be found in Wikipedia:
#https://es.wikipedia.org/wiki/%C3%81rea_metropolitana_de_Madrid

         
# Check if each value is matched with the regular expressions and if so and it is not included in the list of expected cities, then
#will be added to the key.
def audit_city(cit_dict, city_name): 
    m = regex.search(city_name)
    if m:
        cities = m.group()
        if cities not in expected_city:
            cit_dict[cities].add(city_name)

# Check if the element is a city
def is_city_name(elem): 
    return (elem.attrib['k'] == "addr:city")

def audit(osmfile): 
    osm_file = open(osmfile, "r")
    cit_dict = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_city_name(tag):
                    audit_city(cit_dict, tag.attrib['v'])

    return cit_dict

pprint.pprint(dict(audit(file_run))) 

# print the existing city names, so this dict will show me the main issues that I will update and correct in the data cleaning phase.



