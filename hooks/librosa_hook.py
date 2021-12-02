"""A pyinstaller hook required for a package librosa to work."""
from PyInstaller.utils.hooks import collect_all, get_package_paths

datas, binaries, hiddenimports = collect_all("librosa")
print(get_package_paths("librosa"))
