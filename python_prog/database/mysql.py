import mysql.connecter

conn = mysql.connector.connect(host='127.0.0')

cursor = conn.cursor()

cursor.execute(
    'CREATE DATABASE test_mysql_database'
)

cursor.close()
conn.close()

conn = mysql.connecter.connect(host='127.0.0.1', database='test_mysql_database')
cursor = conn.cursor()
cursor.execute('CREATE TABLE persons('
               'id int NOT NULL AUTO_INCREMENT,'
               'name varchar(14) NOT NULL,'
               'PRIMARY KEY(id)')

cursor.execute('INSERT INTO person(name) values("Mike")')
for row in cursor:
    print(row)