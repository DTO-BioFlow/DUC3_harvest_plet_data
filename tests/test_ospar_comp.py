import harvest_plet
print(harvest_plet.__version__)
from harvest_plet import ospar_comp
comp_regions = ospar_comp.OSPARRegions()
id_list = comp_regions.get_all_ids()

# print all items
for item in id_list:
    print(item)

# plot a region
comp_regions.plot_map("SNS")

# plot all regions
comp_regions.plot_map()

# test WKT
my_wkt = comp_regions.get_wkt("SNS")
print(my_wkt)