"""
Main application window for black hole visualization.
"""
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QRadioButton, QSlider, 
                           QLineEdit, QLabel, QVBoxLayout, QFormLayout, QHBoxLayout, QWidget, 
                           QComboBox, QCheckBox, QGridLayout, QDesktopWidget, QFileDialog,
                           QGroupBox)
from PyQt5.QtGui import QImage, QPixmap, QDoubleValidator, QIntValidator, QVector3D, QFont
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt
import sys
import numpy as np
import math
import os

from ..renderers.disk_renderer import render_disk, render_disk_adaptive
from ..renderers.webcam_renderer import render_webcam_loop, render_webcam_loop_adaptive
from ..renderers.image_renderer import render_milky_way, render_milky_way_adaptive
from ..core.utils import create_colormap, map_redshift_to_color_log_g, show_redshift_contours_log, create_redshift_histogram
from ..core.integrators import get_available_methods
# Import configuration functions
from ..config import get_default_config, load_config_from_file, save_config_to_file

class BlackHoleVisualizerWindow(QMainWindow):
    def __init__(self):
        super(BlackHoleVisualizerWindow, self).__init__()
        self.setGeometry(50, 50, 800, 600)
        self.setWindowTitle("Black Hole Raytracer")

        self.inclination = 80
        self.r = 12
        self.angle = math.pi/2 - math.radians(self.inclination)
        self.image = None
        self.history = None
        self.plot_widget = None
        
        self.colormap = 'hot'
        self.grayscale_values, self.colored_values = create_colormap(self.colormap)
        
        self.setup_ui()

    def setup_ui(self):
        """Set up the main user interface"""
        self.widget = QWidget()
        self.layout = QHBoxLayout()
        flo = QFormLayout()
        v_layout = QVBoxLayout()

        run_btn = QPushButton("Render Disk")
        run_btn.clicked.connect(self.run_integration)
        v_layout.addWidget(run_btn)

        webcam_layout = QHBoxLayout()
        webcam_btn = QPushButton("Webcam")
        webcam_btn.clicked.connect(self.run_webcam)
        webcam_layout.addWidget(webcam_btn)

        webcam_btn2 = QPushButton("Preconfigured Webcam")
        webcam_btn2.clicked.connect(self.run_webcam_nice)
        webcam_layout.addWidget(webcam_btn2)

        v_layout.addLayout(webcam_layout)

        milky_btn = QPushButton("Milky Way")
        milky_btn.clicked.connect(self.run_milky)
        v_layout.addWidget(milky_btn)
        
        config_layout = QHBoxLayout()
        
        save_config_btn = QPushButton("Save Config")
        save_config_btn.clicked.connect(self.save_config)
        config_layout.addWidget(save_config_btn)
        
        load_config_btn = QPushButton("Load Config")
        load_config_btn.clicked.connect(self.load_config)
        config_layout.addWidget(load_config_btn)
        
        v_layout.addLayout(config_layout)

        self.radiobutton1 = QRadioButton("Render point slightly before plane")
        self.radiobutton1.setChecked(True)
        self.radiobutton1.planecheck = "False"
        v_layout.addWidget(self.radiobutton1)

        self.radiobutton2 = QRadioButton("Render point in plane")
        self.radiobutton2.planecheck = "True"
        v_layout.addWidget(self.radiobutton2)

        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setMinimum(3)
        self.slider1.setMaximum(15)
        self.slider1.setValue(3)
        self.slider1.setTickPosition(QSlider.TicksBelow)
        self.slider1.setTickInterval(3)
        self.slider1.valueChanged.connect(self.slider1_value_changed)
        self.value_label1 = QLabel()
        self.value_label1.setText(f"Radius inner: {3}\nMin: {3}, Max: {15}")
        v_layout.addWidget(self.value_label1)
        v_layout.addWidget(self.slider1)

        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setMinimum(4)
        self.slider2.setMaximum(30)
        self.slider2.setValue(8)
        self.slider2.setTickPosition(QSlider.TicksBelow)
        self.slider2.setTickInterval(4)
        self.slider2.valueChanged.connect(self.slider2_value_changed)
        self.value_label2 = QLabel()
        self.value_label2.setText(f"Radius outer: {8}\nMin: {4}, Max: {32}")
        v_layout.addWidget(self.value_label2)
        v_layout.addWidget(self.slider2)

        self.slider3 = QSlider(Qt.Horizontal)
        self.slider3.setMinimum(0)
        self.slider3.setMaximum(90)
        self.slider3.setValue(80)
        self.slider3.setTickPosition(QSlider.TicksBelow)
        self.slider3.setTickInterval(5)
        self.slider3.valueChanged.connect(self.slider3_value_changed)
        self.value_label3 = QLabel()
        self.value_label3.setText(f"Inclination: {80}\nMin: {0}, Max: {90}")
        v_layout.addWidget(self.value_label3)
        v_layout.addWidget(self.slider3)

        self.slider4 = QSlider(Qt.Horizontal)
        self.slider4.setMinimum(0)
        self.slider4.setMaximum(400)
        self.slider4.setValue(120)
        self.slider4.setTickPosition(QSlider.TicksBelow)
        self.slider4.setTickInterval(40)
        self.slider4.valueChanged.connect(self.slider4_value_changed)
        self.value_label4 = QLabel()
        self.value_label4.setText(f"Distance: {12}\nMin: {0}, Max: {40}")
        v_layout.addWidget(self.value_label4)
        v_layout.addWidget(self.slider4)

        self.redshiftlines = QCheckBox("Show redshift lines")
        self.redshiftlines.stateChanged.connect(self.run_integration)
        v_layout.addWidget(self.redshiftlines)

        self.traces = QCheckBox("Show traces")
        self.traces.stateChanged.connect(self.run_integration)
        v_layout.addWidget(self.traces)

        self.colors = QComboBox()
        self.colors.addItems([
            "gnuplot2", "hot", "afmhot", "RdBu", "RdBu_r", "inferno", "plasma", "magma", "RdYlBu", "coolwarm", 
            "bwr", "seismic", "twilight", "hsv", "gist_rainbow", "rainbow"
        ])
        self.colors.currentTextChanged.connect(self.set_colors)
        v_layout.addWidget(QLabel("Color mappings"))
        v_layout.addWidget(self.colors)

        self.intmethod = QComboBox()
        self.intmethod.addItems(get_available_methods())
        v_layout.addWidget(QLabel("Integration Method"))
        v_layout.addWidget(self.intmethod)
        
        self.timestep = QLineEdit()
        self.timestep.setValidator(QDoubleValidator(0.00001, 99.99, 5))
        self.timestep.setText(str(0.08))
        self.timestep.editingFinished.connect(self.enter_press)
        
        self.maxsteps = QLineEdit()
        self.maxsteps.setValidator(QIntValidator())
        self.maxsteps.setMaxLength(5)
        self.maxsteps.setText(str(400))
        self.maxsteps.editingFinished.connect(self.enter_press)

        self.samplerate = QLineEdit()
        self.samplerate.setValidator(QIntValidator())
        self.samplerate.setMaxLength(5)
        self.samplerate.setText(str(50))
        self.samplerate.editingFinished.connect(self.enter_press)
        
        self.iwidth = QLineEdit()
        self.iwidth.setValidator(QIntValidator())
        self.iwidth.setMaxLength(5)
        self.iwidth.setText(str(500))
        self.iwidth.editingFinished.connect(self.width_changed)
        
        # self.iheight = QLineEdit()
        # self.iheight.setValidator(QIntValidator())
        # self.iheight.setMaxLength(5)
        # self.iheight.setText(str(500))
        # self.iheight.editingFinished.connect(self.height_changed)
        
        self.fov = QLineEdit()
        self.fov.setValidator(QIntValidator())
        self.fov.setMaxLength(2)
        self.fov.setText(str(90))
        self.fov.editingFinished.connect(self.enter_press)

        adaptive_group = self.setup_adaptive_controls()
        v_layout.addWidget(adaptive_group)

        flo.addRow(v_layout)
        flo.addRow("Time step", self.timestep)
        flo.addRow("Max steps", self.maxsteps)
        flo.addRow("Sample rate for traces", self.samplerate)
        flo.addRow("Image width/height", self.iwidth)
        # flo.addRow("Image height", self.iwidth)
        flo.addRow("Field of view", self.fov)
        self.layout.addLayout(flo)
        
        self.view_window = QLabel()
        self.view_window.setFixedSize(500, 500)
        self.layout.addWidget(self.view_window)

        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.center_window()
        
    def center_window(self):
        """Center the window on the screen"""
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def run_integration(self):
        """Run the integration for black hole with accretion disk"""
        try:
            radius_inner = self.slider1.value()
            radius_outer = self.slider2.value()
            self.inclination = self.slider3.value()
            timestep = float(self.timestep.text())
            maxsteps = int(self.maxsteps.text())
            width = int(self.iwidth.text())
            height = int(self.iwidth.text())
            self.view_window.setFixedSize(width, height)
            fov = int(self.fov.text())

            use_adaptive = self.use_adaptive.isChecked()
            min_h_factor = float(self.min_h_factor.text())
            max_h_factor = float(self.max_h_factor.text()) 
            adapt_threshold = float(self.adapt_threshold.text())
            
            integrator_name = self.intmethod.currentText()

            if self.view_window.width() != width or self.view_window.height() != height:
                self.view_window.setFixedSize(width, height)
                self.center_and_resize_window()

            self.r = self.slider4.value() / 10.0
            self.angle = math.pi/2 - math.radians(self.inclination)
            # self.angle = math.radians(self.inclination)
            x = self.r * math.cos(self.angle)
            y = self.r * math.sin(self.angle)
            camera_position = [0.0, y, x]
            used_points = 0
            
            print(f"Rendering disk: radius inner={radius_inner}, radius outer={radius_outer}, "
            f"inclination={self.inclination}, distance={self.r}, x={x}, y={y}, "
            f"integrator={integrator_name}")

            thread_configs = [(16, 16), (8, 8), (4, 4)] if use_adaptive else [(32, 32), (16, 16), (8, 8)]
            
            for thread_config in thread_configs:
                try:
                    if use_adaptive:
                        self.image, self.history = render_disk_adaptive(
                            width, height, 
                            radius_inner, radius_outer, 
                            camera_position, 
                            timestep, maxsteps, fov,
                            integrator_name,
                            self.traces.isChecked(),
                            int(self.samplerate.text()),
                            min_h_factor,
                            max_h_factor,
                            adapt_threshold
                        )
                        break
                    else:
                        self.image, self.history, used_points = render_disk(
                            width, height, 
                            radius_inner, radius_outer, 
                            camera_position, 
                            timestep, maxsteps, fov,
                            integrator_name,
                            self.traces.isChecked(),
                            int(self.samplerate.text())
                        )
                        break
                except Exception as e:
                    if "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES" in str(e):
                        print(f"Resource error with thread config {thread_config}. Trying a smaller configuration...")
                        if thread_config == thread_configs[-1]:
                            print("All thread configurations failed. Try reducing the image size or turning off adaptive stepping.")
                            print("Rendering failed: CUDA resource error. Try reducing image size or using non-adaptive mode.")
                            return
                    else:
                        print(f"Error with {integrator_name}: {e}")
                        if integrator_name != "Runge-Kutta 4":
                            print("Falling back to Runge-Kutta 4")
                            integrator_name = "Runge-Kutta 4"
                            continue
                        else:
                            print("Rendering failed. Try different parameters.")
                            return
            
            if self.redshiftlines.isChecked():
                # show_redshift_contours(self.image)
                show_redshift_contours_log(self.image, b_or_o=self.intmethod.currentText() in ["Bowie", "Obrechkoff"])
                
            self.load_image()
            self.toggle_gl_view()

            if self.traces.isChecked() and self.history is not None:
                try:
                    self.draw_traces()
                except Exception as e:
                    print(f"Error displaying visualization: {e}")
                
        except Exception as e:
            print(f"Error in integration: {e}")
            import traceback
            traceback.print_exc()

    def run_webcam(self):
        """Run the webcam with black hole distortion"""
        use_adaptive = self.use_adaptive.isChecked()
        if use_adaptive:
            try:
                width = int(self.iwidth.text())
                height = int(self.iwidth.text())
                timestep = float(self.timestep.text())
                maxsteps = int(self.maxsteps.text())
                integrator_name = self.intmethod.currentText()

                min_h_factor = float(self.min_h_factor.text())
                max_h_factor = float(self.max_h_factor.text()) 
                adapt_threshold = float(self.adapt_threshold.text())
                
                self.r = self.slider4.value()
                self.angle = math.pi/2 - math.radians(self.inclination)
                x = self.r * math.cos(self.angle)
                y = self.r * math.sin(self.angle)
                camera_position = [0.0, y, x]
                
                checked = self.radiobutton2.isChecked()

                render_webcam_loop_adaptive(
                    width, height, camera_position, timestep, maxsteps, checked, 
                    integrator_name, min_h_factor, max_h_factor, adapt_threshold
                )
            except Exception as e:
                print(f"Error in webcam: {e}")
                import traceback
                traceback.print_exc()
        else:
            try:
                width = int(self.iwidth.text())
                height = int(self.iwidth.text())
                timestep = float(self.timestep.text())
                maxsteps = int(self.maxsteps.text())
                integrator_name = self.intmethod.currentText()
                
                self.r = self.slider4.value()
                self.angle = math.pi/2 - math.radians(self.inclination)
                x = self.r * math.cos(self.angle)
                y = self.r * math.sin(self.angle)
                camera_position = [0.0, y, x]
                
                checked = self.radiobutton2.isChecked()

                render_webcam_loop(width, height, camera_position, timestep, maxsteps, checked, integrator_name)
            except Exception as e:
                print(f"Error in webcam: {e}")
                import traceback
                traceback.print_exc()

    def run_webcam_nice(self):
        """Run a nice webcam setup"""
        use_adaptive = self.use_adaptive.isChecked()
        width = 1000
        height = 1000
        timestep = 0.01
        maxsteps = 2500
        integrator_name = "Runge-Kutta 4"
        self.r = 9
        self.angle = math.pi/2 - math.radians(85)
        x = self.r * math.cos(self.angle)
        y = self.r * math.sin(self.angle)
        camera_position = [0.0, y, x]
        checked = self.radiobutton2.isChecked()

        if use_adaptive:
            try:
                min_h_factor = float(self.min_h_factor.text())
                max_h_factor = float(self.max_h_factor.text()) 
                adapt_threshold = float(self.adapt_threshold.text())

                self.image = render_webcam_loop_adaptive(
                    width, height, camera_position, timestep, maxsteps, checked,
                    integrator_name, min_h_factor, max_h_factor, adapt_threshold
                )
            except Exception as e:
                print(f"Error in Milky Way rendering: {e}")
                import traceback
                traceback.print_exc()
        else:
            try:
                render_webcam_loop(width, height, camera_position, timestep, maxsteps, checked, integrator_name)
            except Exception as e:
                print(f"Error in webcam: {e}")
                import traceback
                traceback.print_exc()

    def run_milky(self):
        """Run the Milky Way rendering with black hole distortion"""

        use_adaptive = self.use_adaptive.isChecked()

        if use_adaptive:
            try:
                width = int(self.iwidth.text())
                height = int(self.iwidth.text())
                timestep = float(self.timestep.text())
                maxsteps = int(self.maxsteps.text())
                integrator_name = self.intmethod.currentText()
                
                min_h_factor = float(self.min_h_factor.text())
                max_h_factor = float(self.max_h_factor.text()) 
                adapt_threshold = float(self.adapt_threshold.text())
                
                self.r = self.slider4.value()
                self.angle = math.pi/2 - math.radians(self.inclination)
                x = self.r * math.cos(self.angle)
                y = self.r * math.sin(self.angle)
                camera_position = [0.0, y, x]
                
                checked = self.radiobutton2.isChecked()

                self.image = render_milky_way_adaptive(
                    width, height, camera_position, timestep, maxsteps, checked,
                    integrator_name, min_h_factor, max_h_factor, adapt_threshold
                )
                if self.image is not None:
                    self.load_image_external()
            except Exception as e:
                print(f"Error in Milky Way rendering: {e}")
                import traceback
                traceback.print_exc()
        else:
            try:
                width = int(self.iwidth.text())
                height = int(self.iwidth.text())
                timestep = float(self.timestep.text())
                maxsteps = int(self.maxsteps.text())
                integrator_name = self.intmethod.currentText()
                
                self.r = self.slider4.value()
                self.angle = math.pi/2 - math.radians(self.inclination)
                x = self.r * math.cos(self.angle)
                y = self.r * math.sin(self.angle)
                camera_position = [0.0, y, x]
                
                checked = self.radiobutton2.isChecked()

                self.image = render_milky_way(width, height, camera_position, timestep, maxsteps, checked,
                                            integrator_name)
                if self.image is not None:
                    self.load_image_external()
            except Exception as e:
                print(f"Error in Milky Way rendering: {e}")
                import traceback
                traceback.print_exc()

    def slider1_value_changed(self):
        """Handler for inner radius slider changes"""
        current_value = self.slider1.value()
        min_value = self.slider1.minimum()
        max_value = self.slider1.maximum()
        self.value_label1.setText(f"Radius inner: {current_value}\nMin: {min_value}, Max: {max_value}")
        self.run_integration()

    def slider2_value_changed(self):
        """Handler for outer radius slider changes"""
        current_value = self.slider2.value()
        min_value = self.slider2.minimum()
        max_value = self.slider2.maximum()
        self.value_label2.setText(f"Radius outer: {current_value}\nMin: {min_value}, Max: {max_value}")
        self.run_integration()

    def slider3_value_changed(self):
        """Handler for inclination slider changes"""
        current_value = self.slider3.value()
        min_value = self.slider3.minimum()
        max_value = self.slider3.maximum()
        self.value_label3.setText(f"Inclination: {current_value}\nMin: {min_value}, Max: {max_value}")
        self.run_integration()
    
    def slider4_value_changed(self):
        """Handler for distance slider changes"""
        current_value = self.slider4.value() / 10.0
        min_value = self.slider4.minimum()
        max_value = self.slider4.maximum()
        self.value_label4.setText(f"Distance: {current_value:.1f}\nMin: {min_value/10:.0f}, Max: {max_value/10:.0f}")
        self.run_integration()
        
    def set_colors(self, colormap='hot'):
        """Configure color mapping for rendering"""
        self.colormap = colormap
        self.grayscale_values, self.colored_values = create_colormap(colormap)

    def enter_press(self):
        """Handler for when Enter is pressed in input fields"""
        self.run_integration()
    
    def load_image(self):
        """Load and display the redshift image"""
        b_or_o = self.intmethod.currentText() not in ["Bowie", "Obrechkoff"]
        qimage = map_redshift_to_color_log_g(self.image, b_or_o, self.colormap)
        if qimage is not None:
            pixmap = QPixmap.fromImage(qimage)
            self.view_window.setPixmap(pixmap)
        
    def load_image_external(self):
        """Load and display external image (webcam or Milky Way)"""
        if self.image is None:
            print("Error: No image data to display")
            return
            
        height, width, channels = self.image.shape
        bytes_per_line = channels * width
        qimage = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.view_window.setPixmap(pixmap)

    def update_image_dimensions(self):
        """Update the view window size when image dimensions change"""
        try:
            width = int(self.iwidth.text())
            height = int(self.iwidth.text())

            self.view_window.setFixedSize(width, height)
            self.center_and_resize_window()
            
            print(f"Updated display dimensions to: {width}x{height}")
        except Exception as e:
            print(f"Error updating dimensions: {e}")

    def width_changed(self):
        """Handler for width input changes"""
        self.update_image_dimensions()
        
    def height_changed(self):
        """Handler for height input changes"""
        self.update_image_dimensions()

    def center_and_resize_window(self):
        """Center and resize the window based on content"""
        self.adjustSize()
        # self.center_window()
    
    def get_current_config(self):
        """
        Get the current configuration from UI elements
        
        Returns:
            dict: Current configuration settings
        """
        config = {
            'width': int(self.iwidth.text()),
            'height': int(self.iwidth.text()),
            'fov': int(self.fov.text()),
            
            'inner_radius': self.slider1.value(),
            'outer_radius': self.slider2.value(),
            'inclination': self.slider3.value(),
            'distance': self.slider4.value(),
            
            'timestep': float(self.timestep.text()),
            'maxsteps': int(self.maxsteps.text()),
            
            'colormap': self.colors.currentText(),
            'integration_method': self.intmethod.currentText(),
            
            'show_redshift_lines': self.redshiftlines.isChecked(),
            'show_traces': self.traces.isChecked(),
            'render_in_plane': self.radiobutton2.isChecked(),

            'use_adaptive': self.use_adaptive.isChecked(),
            'min_h_factor': float(self.min_h_factor.text()),
            'max_h_factor': float(self.max_h_factor.text()),
            'adapt_threshold': float(self.adapt_threshold.text())
        }
        
        return config
    
    def apply_config(self, config):
        """
        Apply configuration settings to UI elements
        
        Args:
            config (dict): Configuration settings to apply
        """
        try:
            self.iwidth.setText(str(config.get('width', 500)))
            self.iwidth.setText(str(config.get('height', 500)))
            self.fov.setText(str(config.get('fov', 90)))
            
            self.slider1.setValue(config.get('inner_radius', 3))
            self.slider2.setValue(config.get('outer_radius', 8))
            self.slider3.setValue(config.get('inclination', 80))
            self.slider4.setValue(config.get('distance', 12))

            self.timestep.setText(str(config.get('timestep', 0.08)))
            self.maxsteps.setText(str(config.get('maxsteps', 400)))
            
            colormap = config.get('colormap', 'afmhot')
            index = self.colors.findText(colormap)
            if index >= 0:
                self.colors.setCurrentIndex(index)
                
            integration_method = config.get('integration_method', 'Runge-Kutta 4')
            index = self.intmethod.findText(integration_method)
            if index >= 0:
                self.intmethod.setCurrentIndex(index)
                
            self.redshiftlines.setChecked(config.get('show_redshift_lines', False))
            self.traces.setChecked(config.get('show_traces', False))
            
            render_in_plane = config.get('render_in_plane', False)
            if render_in_plane:
                self.radiobutton2.setChecked(True)
            else:
                self.radiobutton1.setChecked(True)
            
            self.use_adaptive.setChecked(config.get('use_adaptive', True))
            self.min_h_factor.setText(str(config.get('min_h_factor', 0.1)))
            self.max_h_factor.setText(str(config.get('max_h_factor', 2.0)))
            self.adapt_threshold.setText(str(config.get('adapt_threshold', 0.01)))
            self.toggle_adaptive_controls()

            self.update_image_dimensions()
            
            print("Configuration successfully applied")
        except Exception as e:
            print(f"Error applying configuration: {e}")
    
    def save_config(self):
        """Save current configuration to a file"""
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Configuration", 
                os.path.expanduser("~"), 
                "JSON Files (*.json);;All Files (*)", 
                options=options
            )
            
            if not file_path:
                return
    
            if not file_path.endswith('.json'):
                file_path += '.json'

            config = self.get_current_config()
            save_config_to_file(config, file_path)
            
            print(f"Configuration saved to: {file_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def load_config(self):
        """Load configuration from a file"""
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Load Configuration", 
                os.path.expanduser("~"), 
                "JSON Files (*.json);;All Files (*)", 
                options=options
            )
            
            if not file_path:
                return

            config = load_config_from_file(file_path)
            self.apply_config(config)
            # self.run_integration()
            
            print(f"Configuration loaded from: {file_path}")
        except Exception as e:
            print(f"Error loading configuration: {e}")

    def toggle_gl_view(self):
        if self.traces.isChecked():
            if self.plot_widget is None:
                self.plot_widget = gl.GLViewWidget()
                self.plot_widget.setFixedSize(int(self.iwidth.text()), int(self.iwidth.text()))
                self.xgrid = gl.GLGridItem()
                self.ygrid = gl.GLGridItem()
                self.zgrid = gl.GLGridItem()
                self.plot_widget.addItem(self.xgrid)
                self.plot_widget.addItem(self.ygrid)
                self.plot_widget.addItem(self.zgrid)
                self.layout.addWidget(self.plot_widget)
        else:
            if self.plot_widget is not None:
                self.plot_widget.clear()
                self.layout.removeWidget(self.plot_widget)
                self.plot_widget.hide()
                self.plot_widget.deleteLater()
                self.plot_widget = None

    def draw_traces(self):
        self.clear_plot_items()

        width = int(self.iwidth.text())
        height = int(self.iwidth.text())
        maxsteps = int(self.maxsteps.text())
        
        self.plot_widget.setBackgroundColor((255, 255, 255, 255))
        
        axis = gl.GLAxisItem(size=QVector3D(10, 10, 10))
        self.plot_widget.addItem(axis)
        
        x_label = gl.GLTextItem(pos=(10, 0, 0), text="X", color=(0, 0, 0, 255))
        y_label = gl.GLTextItem(pos=(0, 10, 0), text="Y", color=(0, 0, 0, 255))
        z_label = gl.GLTextItem(pos=(0, 0, 10), text="Z", color=(0, 0, 0, 255))
        self.plot_widget.addItem(x_label)
        self.plot_widget.addItem(y_label)
        self.plot_widget.addItem(z_label)
        
        colors = np.array([
            (0, 0, 255, 50),
            (0, 128, 255, 150),
            (255, 0, 0, 255)
        ])
        
        samplerate = int(self.samplerate.text())
        
        self.line_items = []
        for x in range(width):
            for y in range(height):
                if x % samplerate == 0 and y % samplerate == 0:
                    idx = int((y // samplerate) * math.ceil(width/samplerate) + (x // samplerate))
                    
                    valid_points = []
                    for i in range(len(self.history[idx])):
                        point = self.history[idx][i]
                        if not (point[0] == 0 and point[1] == 0 and point[2] == 0):
                            valid_points.append(point)
                    
                    if len(valid_points) < 2:
                        continue
                    
                    points_array = np.array(valid_points)
                    n_points = len(points_array)
                    
                    trajectory_colors = np.zeros((n_points, 4))
                    
                    for i in range(n_points):
                        t = i / (n_points - 1)
                        if t < 0.5:
                            s = t * 2
                            c = colors[0] * (1 - s) + colors[1] * s
                        else:
                            s = (t - 0.5) * 2
                            c = colors[1] * (1 - s) + colors[2] * s
                        
                        trajectory_colors[i] = c / 255
                    
                    drawing_variable = gl.GLLinePlotItem(
                        pos=points_array,
                        width=2.5,
                        color=trajectory_colors,
                        antialias=True
                    )
                    
                    self.plot_widget.addItem(drawing_variable)
                    self.line_items.append(drawing_variable)
        
        grid = gl.GLGridItem()
        grid.setSize(10, 10, 0)
        grid.setSpacing(1, 1, 1)
        grid.setColor((100, 100, 100, 100))
        self.plot_widget.addItem(grid)
        
        for i in range(self.slider2.value()):
            tick_label = gl.GLTextItem(pos=(i, -0.5, 0), text=str(i), color=(0, 0, 0, 255))
            self.plot_widget.addItem(tick_label)

        # legend_start = gl.GLTextItem(pos=(-8, -8, 0), text="Start", color=(0, 0, 255, 255))
        # legend_end = gl.GLTextItem(pos=(-8, -7, 0), text="End", color=(255, 0, 0, 255))
        # self.plot_widget.addItem(legend_start)
        # self.plot_widget.addItem(legend_end)

    def clear_plot_items(self):
        if hasattr(self, 'line_items'):
            for item in self.line_items:
                self.plot_widget.removeItem(item)
            self.line_items = []

    def setup_adaptive_controls(self):
        self.adaptive_group = QGroupBox("Adaptive Step Size")
        adaptive_layout = QVBoxLayout()
        
        self.use_adaptive = QCheckBox("Use Adaptive Step Size")
        self.use_adaptive.setChecked(True)
        self.use_adaptive.stateChanged.connect(self.toggle_adaptive_controls)
        adaptive_layout.addWidget(self.use_adaptive)
        
        param_layout = QFormLayout()
        
        self.min_h_factor = QLineEdit()
        self.min_h_factor.setValidator(QDoubleValidator(0.001, 0.5, 3))
        self.min_h_factor.setText("0.1")
        self.min_h_factor.editingFinished.connect(self.enter_press)
        param_layout.addRow("Min Step Factor:", self.min_h_factor)
        
        self.max_h_factor = QLineEdit()
        self.max_h_factor.setValidator(QDoubleValidator(1.0, 5.0, 3))
        self.max_h_factor.setText("2.0")
        self.max_h_factor.editingFinished.connect(self.enter_press)
        param_layout.addRow("Max Step Factor:", self.max_h_factor)
        
        self.adapt_threshold = QLineEdit()
        self.adapt_threshold.setValidator(QDoubleValidator(0.001, 0.5, 3))
        self.adapt_threshold.setText("0.01")
        self.adapt_threshold.editingFinished.connect(self.enter_press)
        param_layout.addRow("Adapt Threshold:", self.adapt_threshold)
        
        adaptive_layout.addLayout(param_layout)
        
        self.adaptive_status = QLabel("Adaptive stepping is enabled")
        adaptive_layout.addWidget(self.adaptive_status)
        
        self.adaptive_group.setLayout(adaptive_layout)
        
        return self.adaptive_group

    def toggle_adaptive_controls(self):
        """Enable or disable adaptive step size controls based on checkbox state"""
        enabled = self.use_adaptive.isChecked()
        
        self.min_h_factor.setEnabled(enabled)
        self.max_h_factor.setEnabled(enabled)
        self.adapt_threshold.setEnabled(enabled)
        
        if enabled:
            self.adaptive_status.setText("Adaptive stepping is enabled")
        else:
            self.adaptive_status.setText("Using fixed step size")
        
        if hasattr(self, 'image'):
            self.run_integration()


