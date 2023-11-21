from vrp.data_utils import InstanceBlueprint

dataset = {}
dataset['0'] = InstanceBlueprint(nb_customers=346, depot_position='C', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=1, demand_max=10, capacity=180, grid_size=1000)  # PA.py

dataset['1'] = InstanceBlueprint(nb_customers=99, depot_position='C', customer_position='R', nb_customer_cluster=None,
    demand_type='inter', demand_min=1, demand_max=10, capacity=180, grid_size=1000)  # PA.py
