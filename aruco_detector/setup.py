from setuptools import setup

package_name = 'aruco_detector'

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
    maintainer='zhouyufei',
    maintainer_email='zhouyufei@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'aruco_detector = aruco_detector.aruco_detector_node:main',
        'aruco_detector_deep = aruco_detector.aruco_detector_deep_node:main',
        ],
    },
)
