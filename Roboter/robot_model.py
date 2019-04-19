import os
import csv
import collections
import view
import logging


logger = logging.getLogger(__name__)


class Robot(object):
    def __init__(self, name, path="./", file_name="rcnt.csv", color = "green"):
        self.name = name
        self.path = path
        self.file_name = path + file_name
        self.opponent_name = None
        self.color = color
        self.restaurant_list = collections.defaultdict(int)

    def hello(self):
        hello_template = view.get_template("hello.txt", self.color)
        self.opponent_name = input(\
            hello_template.replace("$robot_name", self.name))

    def goodbye(self):
        goodbye_template = view.get_template("goodbye.txt", self.color)
        goodbye = goodbye_template.replace("$opponent_name", self.opponent_name)
        print(goodbye)

    def recommend(self):
        if os.path.exists(self.file_name):
            with open(self.file_name, "r") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    self.restaurant_list[row["Name"]] = int(row["Count"])

        for restaurant, cnt in sorted(self.restaurant_list.items(), key=lambda x: -x[1]):
            recommend_template = view.get_template("recommend.txt", self.color)
            answer = input(\
                recommend_template.replace("$recommend_restraunt", restaurant)).lower()
            if answer == "y":
                self.restaurant_list[restaurant] += 1
                break

    def ask_favorite(self):
        ask_favorite_template = view.get_template("ask_favorite.txt", self.color)
        answer_restaurant = input(\
            ask_favorite_template.replace("$opponent_name", self.opponent_name)).capitalize()
        self.restaurant_list[answer_restaurant] += 1
        sorted(self.restaurant_list, key=self.restaurant_list.get, reverse=True)
        logger.info({
            'action': 'save',
            'csv_file': self.restaurant_list,
            'status': 'run'
        })
        with open(self.file_name, "w") as csv_file:
            fieldnames = ["Name", "Count"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for restaurant, cnt in sorted(self.restaurant_list.items(), key=lambda x: -x[1]):
                if cnt == 0:
                    continue
                writer.writerow({"Name": restaurant, "Count": cnt})
        logger.info({
            'action': 'save',
            'status': 'sucess'
        })
