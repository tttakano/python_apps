

class reservation_manage_system(object):
    def __init__(self, type, name="travel", path="./", filename='travel'):
        self.type = type
        self.name = name
        self.path = path
        self.file_name = path + filename
        #TODO create database and set all data 2

    def check(self, date):
        print("check")

    def view_all(self):
        print('view_all')

    def reservation(self, data):
        print('reservation')