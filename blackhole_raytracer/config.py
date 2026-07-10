"""
Configuration settings for the Black Hole Raytracer.
"""
import json

DEFAULT_CONFIG = {
    # Display settings
    'width': 500,
    'height': 500,
    'fov': 90,
    
    # Physics settings
    'inner_radius': 3,
    'outer_radius': 8,
    'inclination': 80,
    'distance': 12,
    
    # Integration settings
    'timestep': 0.08,
    'maxsteps': 400,
    
    # Visual settings
    'colormap': 'afmhot',
    'integration_method': 'Runge-Kutta 4',
}

def get_default_config():
    """
    Get the default configuration settings.
    
    Returns:
        dict: Default configuration settings
    """
    return DEFAULT_CONFIG.copy()

def load_config_from_file(config_file):
    """
    Load configuration from a file.
    
    Args:
        config_file (str): Path to configuration file
        
    Returns:
        dict: Configuration settings
    """
    config = get_default_config()
    
    try:
        with open(config_file, 'r') as f:
            user_config = json.load(f)
            
        config.update(user_config)
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration")
    
    return config

def save_config_to_file(config, config_file):
    """
    Save configuration to a file.
    
    Args:
        config (dict): Configuration settings
        config_file (str): Path to configuration file
    """
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
            
    except Exception as e:
        print(f"Error saving configuration: {e}")