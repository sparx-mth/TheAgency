from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'internnav_bridge'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.xml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'pyyaml',
        'requests',
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='A configurable ROS2 bridge for InternNav navigation models',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bridge_node = internnav_bridge.bridge_node:main',
            'model_server = internnav_bridge.model_server:main',
            'action_executor = internnav_bridge.action_executor:main',
            'test_client = internnav_bridge.internnav_client:test_client',
        ],
    },
)
