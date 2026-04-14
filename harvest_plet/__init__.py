"""
This is the docstring for the harvest_plet module.
"""

import re
import subprocess
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version


def _version_from_git(root: Path) -> str:
	"""Derive a PEP 440 version from the latest git tag."""
	try:
		result = subprocess.run(
			["git", "describe", "--tags", "--dirty", "--long", "--match", "v[0-9]*"],
			cwd=root,
			check=True,
			capture_output=True,
			text=True,
		)
	except Exception as exc:
		raise RuntimeError("Unable to derive version from git metadata") from exc

	describe = result.stdout.strip()
	match = re.fullmatch(r"v?(?P<tag>\d+(?:\.\d+)*)(?:-(?P<count>\d+)-g(?P<sha>[0-9a-f]+))?(?P<dirty>-dirty)?", describe)
	if not match:
		raise RuntimeError(f"Unexpected git describe output: {describe}")

	tag = match.group("tag")
	count = match.group("count")
	sha = match.group("sha")
	dirty = match.group("dirty")

	if not count or count == "0":
		version_str = tag
	else:
		version_str = f"{tag}.post{count}+g{sha}"

	if dirty:
		version_str = f"{version_str}.dirty"

	return version_str


def _get_version() -> str:
	root = Path(__file__).resolve().parent.parent
	try:
		return version("harvest_plet")
	except PackageNotFoundError:
		try:
			return _version_from_git(root)
		except Exception:
			return "0+unknown"


__version__ = _get_version()

