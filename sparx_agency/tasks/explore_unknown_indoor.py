from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--max_room_number", type=int, default=4)
    return parser.parse_args()

def explore_room(map_file):
    ## call some core algorithm here
    ## update map
    pass

def find_unexplored_rooms(max_room_number):
    """
     Find all unexplored rooms in the map
     Args:
         max_room_number:

     Returns:
         map:

     """
    # call some core algorithm here
    # if found, call explore_room
    # else return an updated map


def save_map():
    """
    Save the updated map to a file
    """
    pass


def main():
    args = parse_args()
    map = find_unexplored_rooms(args.max_room_number)
    save_map()




if __name__ == "__main__":
    main()

