Usage
==================

The code below is also available as Notebooks at `https://github.com/willem0boone/demo_harvest_plet <https://github.com/willem0boone/demo_harvest_plet>`_

OSPAR COMP areas
-----------------
This code demonstrates how to use the ospar_comp functionality in the harvest_plet package. The OSPAR zones are used to perform spatial filters in the PLET requests.

.. code-block:: python

    from harvest_plet import ospar_comp
    comp_regions = ospar_comp.OSPARRegions()


List the IDs of regions, this can be used to select one later.

.. code-block:: python

    id_list = comp_regions.get_all_ids()

    for item in id_list:
        print(item)


Plot single COMP area

.. code-block:: python

    comp_regions.plot_map("SNS")


Plot all COMP areas

.. code-block:: python

    comp_regions.plot_map()


Get WKT string

.. code-block:: python

    my_wkt = comp_regions.get_wkt("SNS")
    print(my_wkt)



PLETHarvester
-------------

Find all dataset names

.. code-block:: python

    dataset_names = plet_harvester.get_dataset_names()

    for i, name in enumerate(dataset_names):
        print(i, name)


Load SNS WKT for spatial filter

.. code-block:: python

    wkt_sns = ospart_regions.get_wkt('SNS', simplify=True)


Define time range

.. code-block:: python

    from datetime import date
    start_date = date(2017, 1, 1)
    end_date = date(2020, 1, 1)


Harvest the data

.. code-block:: python

    harvest = plet_harvester.harvest_dataset(
    start_date=start_date,
    end_date=end_date,
    wkt=wkt_sns,
    dataset_name="BE Flanders Marine Institute (VLIZ) - LW_VLIZ_phyto")



Changing the base URL
---------------------
In 2026 a clone was setup that harvests data from ICES DOME API.
To use this clone, the base URL needs to be changed to
``https://www.dassh.ac.uk/plet-dome/cgi-bin/get_form.py`` instead of
``https://www.dassh.ac.uk/plet/cgi-bin/get_form.py``

Also the dataset names are changed, these are scraped from the website
``SITE_URL": "https://www.dassh.ac.uk/plet-dome/``
instead of
``https://www.dassh.ac.uk/lifeforms/``

The package offers simple selection:

.. code-block:: python

    from harvest_plet.plet import PLETHarvester
    plet_harvester = PLETHarvester()
    pprint(plet_harvester.get_instances())

This will return

.. code-block:: python

    {'PLET':{
            'BASE_URL': 'https://www.dassh.ac.uk/plet/cgi-bin/get_form.py',
            'SITE_URL': 'https://www.dassh.ac.uk/lifeforms/'
            },
    'PLET-DOME':{
        'BASE_URL': 'https://www.dassh.ac.uk/plet-dome/cgi-bin/get_form.py',
               'SITE_URL': 'https://www.dassh.ac.uk/plet-dome/'
                }
    }


You can select an instance by setting instance

.. code-block:: python

    plet_harvester.set_instance("PLET-DOME")


After this the harvesting can be done as before, but now the data will be
harvested from the PLET DOME instance instead of the original PLET instance.


Harvest full data for assessment
--------------------------------

.. code-block:: python

    from datetime import date
    from harvest_plet.harvest_plet import harvest_for_assessment
    start_date = date(2015, 1, 1)
    end_date = date(2025, 1, 1)

    df = harvest_for_assessment(start_date=start_date,
                                end_date=end_date)

    for i, item in enumerate(df.head(10).itertuples()):
        print(i, item)

If you want to define the PLET instance to harvest from, you can set the
instance before calling the function:

.. code-block:: python

    from datetime import date
    from harvest_plet.harvest_plet import harvest_for_assessment
    from harvest_plet.plet import PLETHarvester

    plet = PLETHarvester()
    plet.set_instance("PLET-DOME")

    start_date = date(2020, 1, 1)
    end_date = date(2021, 1, 1)

    df = harvest_for_assessment(start_date=start_date,
                                end_date=end_date,
                                plet_harvester=plet,
                                logs_dir="logs")