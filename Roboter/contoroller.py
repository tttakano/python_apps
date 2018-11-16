from robot_model import Robot


def talking_robot_about_restaurant():
    robot = Robot("roboter")
    robot.hello()
    robot.recommend()
    robot.ask_favorite()
    robot.goodbye()