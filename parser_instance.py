import json
import numpy as np
from pathlib import Path

pa_regions = ['0', '1']
rj_regions = ['0', '1', '2', '3', '4', '5']
df_regions = ['0', '1', '2']
alphas = ['025', '050', '075', '100', '125']

EARTH_RADIUS_METERS = 6371000


def calculate_distance_matrix_great_circle_m(
    points
) -> np.ndarray:
    """Distance matrix using the Great Circle distance
    This is an Euclidean-like distance but on spheres [1]. In this case it is
    used to estimate the distance in meters between locations in the Earth.

    Parameters
    ----------
    points
        Iterable with `lat` and `lng` properties with the coordinates of a
        delivery

    Returns
    -------
    distance_matrix
        Array with the (i, j) entry indicating the Great Circle distance (in
        meters) between the `i`-th and the `j`-th point

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Great-circle_distance
    Using the third computational formula
    """
    points_rad = np.radians([(point['lat'], point['lng']) for point in points])

    delta_lambda = points_rad[:, [1]] - points_rad[:, 1]  # (N x M) lng
    phi1 = points_rad[:, [0]]  # (N x 1) array of source latitudes
    phi2 = points_rad[:, 0]  # (1 x M) array of destination latitudes

    delta_sigma = np.arctan2(
        np.sqrt(
            (np.cos(phi2) * np.sin(delta_lambda)) ** 2
            + (
                np.cos(phi1) * np.sin(phi2)
                - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
            )
            ** 2
        ),
        (
            np.sin(phi1) * np.sin(phi2)
            + np.cos(phi1) * np.cos(phi2) * np.cos(delta_lambda)
        ),
    )

    return EARTH_RADIUS_METERS * delta_sigma


# Helper function to convert latitude and longitude to cartesian coordinates
def convert_to_grid_coords(lat, lng, lat_min, lat_max, lng_min, lng_max, grid_size=1000, depot_position=(500, 500)):
    # Normalizing latitude and longitude
    lat_norm = (lat - lat_min) / (lat_max - lat_min)
    lng_norm = (lng - lng_min) / (lng_max - lng_min)

    # Scaling to the grid size
    x = int(lng_norm * grid_size)
    y = int(lat_norm * grid_size)

    # Translating so that depot is at the specified position
    x += (depot_position[0] - int(grid_size / 2))
    y += (depot_position[1] - int(grid_size / 2))

    return x, y

def convert_json_to_vrp(file_path, output_path=None, saves=True, calculate_real_distance=False):
    # Loading the JSON data
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    # Extracting depot data
    depot = json_data['depot']

    partial_demands = json_data['demands']

    # Finding the min and max latitude and longitude values from the JSON data
    latitudes = [demand['point']['lat'] for demand in partial_demands] + [depot['lat']]
    longitudes = [demand['point']['lng'] for demand in partial_demands] + [depot['lng']]
    lat_min, lat_max = min(latitudes), max(latitudes)
    lng_min, lng_max = min(longitudes), max(longitudes)

    # Converting the depot and demands to VRP format with the new coordinate system
    depot_coords = convert_to_grid_coords(depot['lat'], depot['lng'], lat_min, lat_max, lng_min, lng_max)
    demands = [f"1\t0"]  # Demand at depot is zero

    # Preparing nodes with the new coordinate system
    nodes = [f"1\t{depot_coords[0]}\t{depot_coords[1]}"]  # Depot as the first node
    for i, demand in enumerate(partial_demands, start=2):
        node_coords = convert_to_grid_coords(demand['point']['lat'], demand['point']['lng'], lat_min, lat_max, lng_min, lng_max)
        nodes.append(f"{i}\t{node_coords[0]}\t{node_coords[1]}")
        demands.append(f"{i}\t{demand['size'] if demand['type'] == 'PICKUP' else -demand['size']}")
        # demands.append(f"{i}\t{demand['size']}")

    # Constructing the VRP file content
    vrp_content = f"NAME : {json_data['name']}\n"
    vrp_content += "TYPE : CVRP\n"
    vrp_content += f"DIMENSION : {len(nodes)}\n"
    vrp_content += "EDGE_WEIGHT_TYPE : EUC_2D\n"
    vrp_content += f"CAPACITY : {json_data['vehicle_capacity']}\n"
    vrp_content += "NODE_COORD_SECTION\n"
    vrp_content += '\n'.join(nodes)
    vrp_content += "\nDEMAND_SECTION\n"
    vrp_content += '\n'.join(demands)
    vrp_content += "\nDEPOT_SECTION\n1\n-1\nEOF"

    # Saving the VRP content to a file
    if saves:
        with open(output_path, 'w') as file:
            file.write(vrp_content)
    
    distance = None
    if calculate_real_distance:
        points = [depot] + [d['point'] for d in partial_demands]
        distance = calculate_distance_matrix_great_circle_m(points)

    return vrp_content, distance
