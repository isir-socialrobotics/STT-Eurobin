from setuptools import setup

package_name = 'awarebot'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    maintainer='Javad Amirian',
    maintainer_email='amiryan.j@gmail.com',
    description='Context-aware Social Robot',
    license='Apache License 2.0',
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],

    entry_points={
        'console_scripts': [
            'speech_node = awarebot.speech_node:main',
        ],
    },
)
