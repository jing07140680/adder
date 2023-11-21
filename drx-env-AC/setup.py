from setuptools import setup

setup(
    name="drx_env_AC",
    version="0.0.1",
    install_requires=["gym==0.15.4"],
    kwargs={'debug': False, 'timelineplot': False}  
)
