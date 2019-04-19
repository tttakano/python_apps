import happybase

connection = happybase.Connection('localhost')
connection.open()


# connection.create_table(b'sns', {'blog': dict()})

table = connection.table(b'sns')

# table.put(
#     b'user1', {
#         b'blog:bitcoin': b'user1 about bitcoin',
#         b'blog:soccer': b'user1 about soccer',
#     }
# )
#
# table.put(
#     b'user2', {
#         b'blog:soccer': b'user2 about soccer',
#     }
# )

print(list(table.scan()))
print()
print(list(table.scan(row_prefix=b'user1')))
print()
print(list(table.scan(columns=[b'blog:soccer'])))

connection.disable_table(b'sns')
connection.delete_table(b'sns')