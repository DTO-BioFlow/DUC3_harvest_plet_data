from pprint import pprint
from datetime import date
from harvest_plet.plet import PLETHarvester
from harvest_plet.harvest_for_assessment import harvest_for_assessment
plet_harvester = PLETHarvester()

plet = PLETHarvester()

# See all configured instances
pprint(plet.get_instances())

print("-"*50)

# Switch by key from endpoints.json
plet.set_instance("PLET-DOME")

start_date = date(2020, 1, 1)
end_date = date(2021, 1, 1)

df = harvest_for_assessment(start_date=start_date,
                            end_date=end_date,
                            plet_harvester=plet,
                            logs_dir="logs")

for i, item in enumerate(df.head(10).itertuples()):
    print(i, item)

