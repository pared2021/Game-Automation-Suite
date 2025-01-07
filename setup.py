from setuptools import setup, find_packages

setup(
    name='game_automation_suite',
    version='0.1.0',
    description='A suite for automating game tasks',
    author='pared2021',
    author_email='pared2021@example.com',
    url='https://github.com/pared2021/Game-Automation-Suite',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'pyautogui',
        'pytesseract',
        'Pillow'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)