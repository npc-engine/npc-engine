"""A pyinstaller hook required for a package librosa to work."""
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all("librosa")
