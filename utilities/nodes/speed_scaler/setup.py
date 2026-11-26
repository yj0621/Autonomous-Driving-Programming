from setuptools import setup

package_name = 'speed_scaler'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='misys',
    maintainer_email='jimin1203@soongsil.ac.kr',
    description='Scales the velocity of the global waypoints by a single global factor',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'speed_scaler = speed_scaler.speed_scaler:main',
        ],
    },
)
