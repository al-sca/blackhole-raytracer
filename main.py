"""
Main entry point for the Black Hole Raytracer application.
"""
import sys
from PyQt5.QtWidgets import QApplication

from blackhole_raytracer.gui.main_window import BlackHoleVisualizerWindow

def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    gui = BlackHoleVisualizerWindow()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()