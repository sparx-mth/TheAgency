from setuptools import setup, find_packages

package_name = 'internnav_bridge'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'setuptools',
        'requests',
        'numpy',
        'opencv-python',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='ROS2 bridge for InternNav model server',
    license='MIT',
    entry_points={
        'console_scripts': [
            'bridge_node = internnav_bridge.bridge_node:main',
        ],
    },
)