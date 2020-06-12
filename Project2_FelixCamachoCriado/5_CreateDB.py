#import libraries
import sqlite3
import csv
from pprint import pprint

# Connect to the database (if it doesn't exist, it will be created in the folder that your notebook is in):
sqlite_file = 'Madrid.db'    # name of the sqlite database file

# Connect to the database
conn = sqlite3.connect(sqlite_file)

# Get a cursor object
cur = conn.cursor()

# Drop every existing table

cur.execute(''' DROP TABLE IF EXISTS nodes_tags''')
conn.commit()
cur.execute(''' DROP TABLE IF EXISTS nodes''')
conn.commit()
cur.execute(''' DROP TABLE IF EXISTS ways''')
conn.commit()
cur.execute(''' DROP TABLE IF EXISTS ways_tags''')
conn.commit()
cur.execute(''' DROP TABLE IF EXISTS ways_nodes''')
conn.commit()

#----------------------TABLE NODES-------------------------#
# Create the table nodes, specifying the column names and data types:
cur.execute(''' CREATE TABLE nodes(id INTEGER, lat INTEGER, lon INTEGER, user TEXT, uid INTEGER, version INTEGER, changeset TEXT,
timestamp TEXT)
''')

# commit the changes
conn.commit()

# check if the created structure is correct
cur.execute("PRAGMA TABLE_INFO(nodes)")
all_rows = cur.fetchall()
print('NODES TABLE STRUCTURE:')
pprint(all_rows)

# Read in the csv file as a dictionary, 
with open('nodes.csv','r') as fin:
    dr = csv.DictReader(fin) 
    to_db = [(i['id'], i['lat'], i['lon'], i['user'],i['uid'], i['version'], i['changeset'], i['timestamp']) 
    for i in dr]
    
# insert the formatted data
cur.executemany("INSERT INTO nodes(id, lat, lon, user, uid, version, changeset, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?);", to_db)
# commit the changes
conn.commit()


#----------------------TABLE NODES_TAGS-------------------------#

# Create the table, specifying the column names and data types:
cur.execute(''' CREATE TABLE nodes_tags(id INTEGER , key TEXT, value TEXT, type TEXT) ''')

# commit the changes
conn.commit()
# check if the created structure is correct
cur.execute("PRAGMA TABLE_INFO(nodes_tags)")
all_rows = cur.fetchall()
print('NODES_TAGS TABLE STRUCTURE:')
pprint(all_rows)

# Read in the csv file as a dictionary, 
with open('nodes_tags.csv','r') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['key'], i['value'], i['type']) for i in dr]
    
# insert the formatted data
cur.executemany("INSERT INTO nodes_tags(id, key, value, type) VALUES (?, ?, ?, ?);", to_db)
# commit the changes
conn.commit()

#----------------------TABLE WAYS-------------------------#

# Create the table, specifying the column names and data types:
cur.execute(''' CREATE TABLE ways(id INTEGER, user TEXT, uid INTEGER, version INTEGER, changeset TEXT, timestamp TEXT) ''')

# commit the changes
conn.commit()

# check if the created structure is correct
cur.execute("PRAGMA TABLE_INFO(ways)")
all_rows = cur.fetchall()
print('WAYS TABLE STRUCTURE:')
pprint(all_rows)

# Read in the csv file as a dictionary
with open('ways.csv','r') as fin:
    dr = csv.DictReader(fin) 
    to_db = [(i['id'], i['user'], i['uid'], i['version'], i['changeset'], i['timestamp']) 
               for i in dr]

# insert the formatted data
cur.executemany("INSERT INTO ways(id, user, uid, version, changeset, timestamp) VALUES (?, ?, ?, ?, ?, ?);", to_db)
# commit the changes
conn.commit()

#----------------------TABLE WAYS_NODES-------------------------#

# Create the table, specifying the column names and data types:
cur.execute(''' CREATE TABLE ways_nodes(id INTEGER, node_id INTEGER, position INTEGER) ''')

# commit the changes
conn.commit()

# check if the created structure is correct
cur.execute("PRAGMA TABLE_INFO(ways_nodes)")
all_rows = cur.fetchall()
print('WAYS_NODES TABLE STRUCTURE:')
pprint(all_rows)

# Read in the csv file as a dictionary
with open('ways_nodes.csv','r') as fin:
    dr = csv.DictReader(fin) 
    to_db = [(i['id'], i['node_id'], i['position']) for i in dr]

# insert the formatted data
cur.executemany("INSERT INTO ways_nodes(id, node_id, position) VALUES (?, ?, ?);", to_db)
# commit the changes
conn.commit()

#----------------------TABLE WAYS_TAGS-------------------------#

# Create the table, specifying the column names and data types
cur.execute(''' CREATE TABLE ways_tags(id INTEGER, key INTEGER, value TEXT, type TEXT) ''')

# commit the changes
conn.commit()

# check if the created structure is correct
cur.execute("PRAGMA TABLE_INFO(ways_tags)")
all_rows = cur.fetchall()
print('WAYS TAGS STRUCTURE:')
pprint(all_rows)

# Read in the csv file as a dictionary
with open('ways_tags.csv','r') as fin:
     dr = csv.DictReader(fin)
     to_db = [(i['id'], i['key'], i['value'], i['type']) for i in dr]
 
# insert the formatted data
cur.executemany("INSERT INTO ways_tags(id, key, value, type) VALUES (?, ?, ?, ?);", to_db)
# commit the changes
conn.commit()

# close the connection
conn.close()
