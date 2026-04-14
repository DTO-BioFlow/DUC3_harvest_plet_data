import unittest

from harvest_plet.plet import PLETHarvester


class TestPLETInstances(unittest.TestCase):
    def setUp(self) -> None:
        self.harvester = PLETHarvester()

    def test_get_instances_returns_known_keys(self) -> None:
        instances = self.harvester.get_instances()
        self.assertIn("PLET", instances)
        self.assertIn("PLET-DOME", instances)
        self.assertEqual(
            instances["PLET"]["BASE_URL"],
            "https://www.dassh.ac.uk/plet/cgi-bin/get_form.py",
        )

    def test_default_instance_is_plet(self) -> None:
        self.assertEqual(self.harvester.get_instance(), "PLET")
        self.assertEqual(
            self.harvester.get_base_url(),
            "https://www.dassh.ac.uk/plet/cgi-bin/get_form.py",
        )
        self.assertEqual(
            self.harvester.get_site_url(),
            "https://www.dassh.ac.uk/lifeforms/",
        )

    def test_set_instance_switches_both_urls(self) -> None:
        self.harvester.set_instance("PLET-DOME")
        self.assertEqual(self.harvester.get_instance(), "PLET-DOME")
        self.assertEqual(
            self.harvester.get_base_url(),
            "https://www.dassh.ac.uk/plet-dome/cgi-bin/get_form.py",
        )
        self.assertEqual(
            self.harvester.get_site_url(),
            "https://www.dassh.ac.uk/plet-dome/",
        )

    def test_set_instance_with_unknown_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.harvester.set_instance("NOT-AN-INSTANCE")

    def test_manual_ad_hoc_setters_work(self) -> None:
        custom_base = "https://example.org/base"
        custom_site = "https://example.org/site"
        self.harvester.set_base_url(custom_base)
        self.harvester.set_site_url(custom_site)
        self.assertEqual(self.harvester.get_base_url(), custom_base)
        self.assertEqual(self.harvester.get_site_url(), custom_site)


if __name__ == "__main__":
    unittest.main()

