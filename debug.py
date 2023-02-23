from plugin import AudioLDMPlugin
from tuneflow_devkit import Debugger

if __name__ == "__main__":
    Debugger(plugin_class=AudioLDMPlugin).start()
