Installation Guide
=================

This guide will help you install Game Automation Suite and set up your development environment.

System Requirements
-----------------

* Python 3.8 or higher
* 2GB RAM minimum
* 500MB disk space
* Operating System: Windows/Linux/MacOS

Basic Installation
----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/yourusername/game-automation-suite.git
      cd game-automation-suite

2. Install dependencies:

   .. code-block:: bash

      make install

   Or manually:

   .. code-block:: bash

      pip install -r requirements.txt

Development Installation
----------------------

For development, you'll need additional tools and dependencies:

.. code-block:: bash

   pip install -r requirements-dev.txt
   pre-commit install

Optional Dependencies
-------------------

AI Features:
~~~~~~~~~~~

For AI-related features:

.. code-block:: bash

   pip install -r requirements-ai.txt

NLP Features:
~~~~~~~~~~~~

For natural language processing features:

.. code-block:: bash

   pip install -r requirements-nlp.txt

Configuration
------------

1. Copy and customize configuration files:

   .. code-block:: bash

      cp config/config.yaml.example config/config.yaml
      cp config/game_settings.yaml.example game_settings.yaml

2. Set up environment variables:

   .. code-block:: bash

      cp .env.example .env
      # Edit .env with your settings

Verification
-----------

Verify your installation:

.. code-block:: bash

   make test

This will run the test suite and verify that everything is working correctly.

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

1. **Missing Dependencies**

   .. code-block:: bash

      pip install --upgrade pip
      pip install -r requirements.txt

2. **Permission Issues**

   Use a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # Linux/Mac
      venv\Scripts\activate     # Windows

3. **Import Errors**

   Ensure Python path is set correctly:

   .. code-block:: bash

      export PYTHONPATH=$PYTHONPATH:/path/to/game-automation-suite

Platform-Specific Notes
----------------------

Windows
~~~~~~~

- Install Visual C++ Build Tools
- Use Windows Terminal for better experience
- Consider using WSL2 for Linux-like environment

macOS
~~~~~

- Install Xcode Command Line Tools
- Use Homebrew for additional dependencies

Linux
~~~~~

- Install required system packages:

  .. code-block:: bash

     sudo apt-get update
     sudo apt-get install python3-dev build-essential

Next Steps
---------

- Read the :doc:`quickstart` guide
- Explore the :doc:`user_guide/index`
- Check out the :doc:`contributing` guide if you want to contribute
