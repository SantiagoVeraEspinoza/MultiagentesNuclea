class Pos:
    def __init__(self, x, y):
        self.x = x
        self.y = y

LOAD_POS = Pos(-31210.0, 22570.0)

class truck:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

def READ_SPEED_FUNCTION():
    speed = 1000
    return speed

def see(pos_x, pos_y):
    truck_agent = truck(pos_x, pos_y)
    args = {0:truck_agent}
    return args

def move():
    return

def main(pos_x, pos_y):
    args = see(pos_x, pos_y)
    print(args)
    move()
    return
