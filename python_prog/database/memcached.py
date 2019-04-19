import time
import sqlite3

import memcache


db = memcache.Client(['127.0.0.1:11211'])

# mongodb.set('web_page', 'value1', time=3)
# time.sleep(1)
# print(mongodb.get('web_page'))

# mongodb.set('counter', 0)
# mongodb.incr('counter', 1)
# mongodb.incr('counter', 1)
# mongodb.incr('counter', 1)
# mongodb.incr('counter', 1)
# print(mongodb.get('counter'))

conn = sqlite3.connect('test_sqlite2.mongodb')
curs = conn.cursor()
curs.execute(
    'CREATE TABLE persons('
    'employed_id INTEGER PRIMARY KEY AUTOINCREMENT, name STRING)')
curs.execute('INSERT INTO persons(name) values("Mike")')
conn.commit()
conn.close()

def get_employ_id(name):
    employ_id = db.get(name)
    if employ_id:
        return employ_id
    curs.execute(
        'SELECT * FROM persons WHERE name = "{}"'.format(name)
    )
    person = curs.fetchone()
    if not person:
        raise Exception('No employed')
    employ_id, name = person
    db.set(name, employ_id, time=60)
    return employ_id

print(get_employ_id("Mike"))
