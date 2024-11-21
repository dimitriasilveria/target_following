from setuptools import find_packages, setup

package_name = 'target_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bitdrones',
    maintainer_email='23ldy@queensu.ca',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'commands_node = target_tracking.commands_node:main',
            'send_commands_node = target_tracking.send_commands_node:main'
        ],
    },
)
