import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sensado'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # --- Archivos de simulación y robótica agregados ---
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='verine',
    maintainer_email='asdelvalle@uc.cl',
    description='Modulo de sensado y simulacion para robot desmalezador Capstone',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'rutina_navegacion = sensado.cf_deteccion_troncos:main'
        ],
    },
)