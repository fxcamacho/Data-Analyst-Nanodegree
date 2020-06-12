#import libraries
import csv, sqlite3

#queries
def number_of_nodes():
	result = cur.execute('SELECT COUNT(*) FROM nodes')
	return result.fetchall()

def number_of_ways():
	result = cur.execute('SELECT COUNT(*) FROM ways')
	return result.fetchall()

def number_of_unique_users():
	result = cur.execute('SELECT COUNT(DISTINCT(e.uid)) \
            FROM (SELECT uid FROM nodes UNION ALL SELECT uid FROM ways) e')
	return result.fetchall()
    
def top_contributing_users():
	users = []
	for row in cur.execute('SELECT e.user, COUNT(*) as num \
            FROM (SELECT user FROM nodes UNION ALL SELECT user FROM ways) e \
            GROUP BY e.user \
            ORDER BY num DESC \
            LIMIT 10'):
		users.append(row)
	return users

def onetime_users():
	users = []
	for row in cur.execute('SELECT COUNT(*) \
           FROM (SELECT e.user, COUNT(*) as num \
           FROM (SELECT user FROM nodes UNION ALL SELECT user FROM ways) e \
           GROUP BY e.user \
           HAVING num=1)  u;'):
		users.append(row)
	return users

def common_shops():
	shops = []
	for row in cur.execute("SELECT value, COUNT(*) as num \
         FROM NODES_TAGS as nt \
         WHERE nt.key = 'shop'\
         GROUP BY nt.value \
         ORDER BY num DESC \
         LIMIT 10;"):
		shops.append(row)
	return shops

def clothing_companies():
	companies = []
	for row in cur.execute("SELECT nt1.value, COUNT(nt1.id) as count \
         FROM nodes_tags nt1 \
         JOIN (SELECT id FROM nodes_tags WHERE value = 'clothes') nt2 \
         ON nt1.id = nt2.id \
         WHERE nt1.key = 'name' \
         GROUP BY nt1.value \
         ORDER BY count DESC \
         LIMIT 20;"):
		companies.append(row)
	return companies

def top_streets():
	top_streets = []
	for row in cur.execute("SELECT value, COUNT(*) as num \
         FROM NODES_TAGS as nt \
         WHERE nt.key = 'street'\
         GROUP BY nt.value \
         ORDER BY num DESC \
         LIMIT 5;"):
		top_streets.append(row)
	return top_streets

def postalcodes_types():
	keys = []
	for row in cur.execute("SELECT key, COUNT(*) as num \
         FROM WAYS_TAGS as wt \
         WHERE (wt.value LIKE '%28%') AND (length(wt.value) = 5)\
         GROUP BY wt.key \
         ORDER BY num DESC;"):
		keys.append(row)
	return keys

if __name__ == '__main__':
	
	con = sqlite3.connect("Madrid.db")
	cur = con.cursor()
	
	print("Number of nodes: " , number_of_nodes())
	print("Number of ways: " , number_of_ways())
	print("Number of unique users: " , number_of_unique_users())
	print("Top contributing users: " , top_contributing_users())
	print("Users appearing only once (having 1 post) " , onetime_users())
	print("Which are the most common type of shops in this area? " , common_shops())
	print("What are the clothing companies with more open stores in the Madrid metropolitan area? " ,
	 clothing_companies())	
	print("Which are the TOP 5 Streets with more presence in the dataset? " , top_streets())
	print("Is there more than one kind of key for representing postal codes in the dataset?" , postalcodes_types())