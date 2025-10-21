from harvest_plet.plet import PLETHarvester
plet_harvester = PLETHarvester()

# Find all dataset names
dataset_names = plet_harvester.get_dataset_names()

for i, name in enumerate(dataset_names):
    print(i, name)


# OSPAR regions
from harvest_plet.ospar_comp import OSPARRegions
ospart_regions = OSPARRegions()

# Define region WKT
wkt_sns = ospart_regions.get_wkt('SNS', simplify=True)
print(wkt_sns)

# set a Time range
from datetime import date
start_date = date(2017, 1, 1)
end_date = date(2020, 1, 1)

# Harvest the data
harvest = plet_harvester.harvest_data(
    start_date=start_date,
    end_date=end_date,
    wkt=wkt_sns,
    dataset_name="BE Flanders Marine Institute (VLIZ) - LW_VLIZ_phyto")
print(harvest)
