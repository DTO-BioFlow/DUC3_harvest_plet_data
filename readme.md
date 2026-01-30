# Demo of harvest PLET
The [Plankton Lifeform Extraction Tool (PLET)](https://www.dassh.ac.uk/lifeforms/) is a database that offers monthly aggregated plankton lifeform concentrations. 

Downloads can be made via the webinterface or this python [endpoint](https://www.dassh.ac.uk/lifeforms/docs/automation_guidance.txt).
There is no fully developed API. 

The [harvest_plet](https://github.com/willem0boone/harvest_plet/tree/main) package facilitates the request forming and handling. Extending the .py endpoint, this package handles following aspects:
- listing available datasets 
- forming & evaluating the request url
- some request result in long respons time, the package allow tweaking timeouts & retry config.

## Read the docs
See [https://harvest-plet.readthedocs.io/en/latest/](https://harvest-plet.readthedocs.io/en/latest/)

## Credits
This is part of DTO-Bioflow Demonstrator use case 3: pelagic biodiversity.
<br>
The DTO-Bioflow project is funded by the European Union under the Horizon Europe Programme, [Grant Agreement No. 101112823](https://cordis.europa.eu/project/id/101112823/results).

(c) Willem Boone, 2026.