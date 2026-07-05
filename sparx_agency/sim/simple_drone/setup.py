from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'simple_drone'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Worlds
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        # Models: Explicitly list the SDF file, not the directory
        (os.path.join('share', package_name, 'models', 'simple_drone'), 
            ['models/simple_drone/model.sdf']),
          (os.path.join('share', package_name, 'models', 'target_box'),  glob('models/target_box/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user1',
    maintainer_email='user1@todo.todo',
    description='Simple drone simulation',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [],
    },
)   