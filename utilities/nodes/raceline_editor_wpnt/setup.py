from setuptools import setup

package_name = 'raceline_editor_wpnt'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='WpntArray raceline editor with RViz interaction',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 패키지/모듈:main
            'raceline_editor_wpnt = raceline_editor_wpnt.raceline_editor_wpnt:main',
        ],
    },
)
