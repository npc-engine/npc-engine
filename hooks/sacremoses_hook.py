"""A pyinstaller hook required for a package scaremoses to work."""
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all("sacremoses")
