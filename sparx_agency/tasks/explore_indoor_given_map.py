from argparse import ArgumentParser

from sparx_agency.core.mapping.map import Map


def parse_args():
    parser = ArgumentParser(description="Explore an indoor area given a known map. Update the map")
    parser.add_argument("--map", type=str, default="indoor_map.json", help="Path to the map file")
    return parser.parse_args()

def explore_room(map_file):
    """Explore a room given a map"""
    ## call some core algorithm here
    ## update map
    pass

def find_unexplored_rooms(map: Map) -> Map:
    """
    Find all unexplored rooms in the map
    Args:
        map:

    Returns:
        updated map:

    """
    # call some core algorithm here
    # if found, call explore_room
    # else return an updated map
    return map


def load_map(map_file):
    """Load the map from a file"""
    map = Map.load(map_file)
    return map

def main():
    args = parse_args()
    map = load_map(args.map)
    map = find_unexplored_rooms(map)
    map.save("updated_map.json")
    print(map)





if __name__ == "__main__":
    main()

