from pprint import pprint
from harvest_plet.plet import PLETHarvester
plet_harvester = PLETHarvester()

plet = PLETHarvester()

# See all configured instances
pprint(plet.get_instances())

print("-"*50)

# Switch by key from endpoints.json
plet.set_instance("PLET-DOME")
print(plet.get_base_url())
print(plet.get_site_url())

print("-"*50)
# Ad-hoc/manual override
plet.set_base_url("https://example.org/get_form.py")
plet.set_site_url("https://example.org/site/")
print(plet.get_base_url())
print(plet.get_site_url())
