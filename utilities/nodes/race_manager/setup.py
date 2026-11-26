from setuptools import setup
import os
from glob import glob

package_name = 'race_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jimin1203',
    maintainer_email='janny030224@gmail.com',
    description='The race_manager package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'obstacle_spawner = race_manager.obstacle_spawner:main',
            'scorer = race_manager.scorer:main',
            'report = race_manager.report:main',
            'dump_candidates = race_manager.dump_candidates:main',
            'diagnose = race_manager.diagnose:main',
        ],
    },
)
