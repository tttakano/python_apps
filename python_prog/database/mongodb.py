import datetime

from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['test_database']

stack1 = {
    'name': 'customer1',
    'pip': ['python', 'java', 'go'],
    'info': {'os': 'mac'},
    'data': datetime.datetime.utcnow()
}

stack2 = {
    'name': 'customer2',
    'pip': ['python', 'java'],
    'info': {'os' : 'windows'},
    'data': datetime.datetime.utcnow()
}

db_stacks = db.stacks
# stack_id = db_stacks.insert_one(stack1).inserted_id
# print(stack_id, type(stack_id))
# print("#########################")
# from bson.objectid import ObjectId
# str_stacked_id = '5c0095fc7f3d62f686d3d5fb'
# print(db_stacks.find_one({'_id': ObjectId(str_stacked_id)}))
# print(db_stacks.find_one({'name': 'customer1'}))
# stack_id = db_stacks.insert_one(stack2).inserted_id
# print(stack_id, type(stack_id))
#
# for stack in db_stacks.find():
#     print(stack)
# now = datetime.datetime.utcnow()
# for stack in db_stacks.find({'data': {'$gt': now}}):
#     print(stack)
# db_stacks.find_one_and_update(
#     {'name': 'customer1'}, {'$set': {'name': 'YYY'}})
# print(db_stacks.find_one({'name': 'YYY'}))
db_stacks.delete_one({'name': 'YYY'})
print(db_stacks.find_one({'name': 'YYY'}))