import os

from setuptools import find_packages, setup

package_name = 'neupan_ros'


def share_tree(src):
    """把 src 目录下的所有文件按原目录结构安装到 share/neupan_ros/<src>/ 下。"""
    data_files = []
    for root, _, names in os.walk(src):
        if names:
            data_files.append((
                os.path.join('share', package_name, root),
                [os.path.join(root, name) for name in names],
            ))
    return data_files


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch / 参数 / rviz / DUNE 权重，节点和 launch 都按 share 目录找这些文件
        *share_tree('launch'),
        *share_tree('config'),
        *share_tree('rviz'),
        *share_tree('model'),
    ],
    # NeuPAN 本体（torch、cvxpy 等）是单独的 pip 包，见 README，这里不列
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='han',
    maintainer_email='han@todo.todo',
    description='ROS 2 wrapper of the NeuPAN planner for the wheeltec V550_mec robot',
    license='GPL-3.0-or-later',
    entry_points={
        'console_scripts': [
            'neupan_node = neupan_ros.neupan_node:main',
            'vel_monitor = neupan_ros.vel_monitor:main',
        ],
    },
)
