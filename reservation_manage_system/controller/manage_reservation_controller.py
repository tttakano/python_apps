from model.manage_args import manage_args
from model.reservation_system_model import reservation_manage_system

def manage_reservation_system():
    options = manage_args()
    manage_system = reservation_manage_system(options.type)
    if manage_system.type == 1:
        manage_system.check()
    elif manage_system.type == 2:
        manage_system.view_all()
    elif manage_system.type == 3:
        manage_system.reservation()


