import json

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

def convert_json_to_vrp(file_path, output_path):
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
    with open(output_path, 'w') as file:
        file.write(vrp_content)

def convert_pa_instances():
    pa_regions = ['0', '1']
    alphas = ['025', '050', '075', '100', '125']

    for r in pa_regions:
        for alpha in alphas:
            for i in range(90):
                file_path = f'/Users/cstenico/Documents/shire/tcc-mba/loggibud/data/{alpha}/vrppd-instances-1.0/train/pa-{r}/cvrp-{r}-pa-{i}.json'
                output_path = f'train_instances/PA_{r}/{alpha}/PA_{i}.vrp'
                convert_json_to_vrp(file_path, output_path)

def convert_dev_pa_instances():
    pa_regions = ['0', '1']
    alphas = ['025', '050', '075', '100', '125']

    for r in pa_regions:
        for alpha in alphas:
            for i in range(90, 120):
                file_path = f'/Users/cstenico/Documents/shire/tcc-mba/loggibud/data/{alpha}/vrppd-instances-1.0/dev/pa-{r}/cvrp-{r}-pa-{i}.json'
                output_path = f'test_instances/PA_{r}/{alpha}/PA_{i}.vrp'
                convert_json_to_vrp(file_path, output_path)

def convert_df_instances():
    df_regions = ['0', '1', '2']
    alphas = ['025', '050', '075', '100', '125']

    for r in df_regions:
        for alpha in alphas:
            for i in range(90):
                file_path = f'/Users/cstenico/Documents/shire/tcc-mba/loggibud/data/{alpha}/vrppd-instances-1.0/train/df-{r}/cvrp-{r}-df-{i}.json'
                output_path = f'train_instances/DF_{r}/{alpha}/DF_{i}.vrp'
                convert_json_to_vrp(file_path, output_path)

def convert_dev_df_instances():
    df_regions = ['0', '1', '2']
    alphas = ['025', '050', '075', '100', '125']

    for r in df_regions:
        for alpha in alphas:
            for i in range(90, 120):
                file_path = f'/Users/cstenico/Documents/shire/tcc-mba/loggibud/data/{alpha}/vrppd-instances-1.0/dev/df-{r}/cvrp-{r}-df-{i}.json'
                output_path = f'test_instances/DF_{r}/{alpha}/DF_{i}.vrp'
                convert_json_to_vrp(file_path, output_path)

def convert_rj_instances():
    rj_regions = ['0', '1', '2']
    alphas = ['025', '050', '075', '100', '125']

    for r in rj_regions:
        for alpha in alphas:
            for i in range(90):
                file_path = f'/Users/cstenico/Documents/shire/tcc-mba/loggibud/data/{alpha}/vrppd-instances-1.0/train/rj-{r}/cvrp-{r}-rj-{i}.json'
                output_path = f'train_instances/RJ_{r}/{alpha}/RJ_{i}.vrp'
                convert_json_to_vrp(file_path, output_path)

def convert_dev_rj_instances():
    rj_regions = ['0', '1', '2', '3', '4', '5']
    alphas = ['025', '050', '075', '100', '125']

    for r in rj_regions:
        for alpha in alphas:
            for i in range(90, 120):
                file_path = f'/Users/cstenico/Documents/shire/tcc-mba/loggibud/data/{alpha}/vrppd-instances-1.0/dev/rj-{r}/cvrp-{r}-rj-{i}.json'
                output_path = f'test_instances/RJ_{r}/{alpha}/RJ_{i}.vrp'
                convert_json_to_vrp(file_path, output_path)